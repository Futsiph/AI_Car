import cv2
import numpy as np
import threading
from ultralytics import YOLO

model = YOLO("../runs/detect/circuit_object-17/weights/best.pt")
model.to('cuda')
source = "../data/test3.mp4"
IMAGE_HEIGHT, IMAGE_WIDTH = 0, 0

# Crop if needed
crop_top = 0  # 120
crop_bottom = 0  # 50
crop_left = 0  # 155
crop_rigth = 0  # 170

max_screen_width = 1024
max_screen_height = 576

# Share var
shared_lines = None
shared_detections = None
lock = threading.Lock()


last_valid_lines = None
lost_frames_count = 0
MAX_LOST_FRAMES = 8


smoothed_angle_left = None
smoothed_angle_right = None
ANGLE_SMOOTHING = 0.3
ANGLE_TOLERANCE = 18


clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def mask_out_detections(binary_image, detections, padding=8):
    if detections is None:
        return binary_image

    h, w = binary_image.shape[:2]
    for r in detections:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes.xyxy:
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            cv2.rectangle(binary_image, (x1, y1), (x2, y2), 0, -1)

    return binary_image


def preprocess_for_lines(frame, detections=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = clahe.apply(gray)
    blur = cv2.bilateralFilter(gray_eq, d=7, sigmaColor=50, sigmaSpace=50)

    white = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=-8
    )

    open_kernel = np.ones((3, 3), np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, open_kernel)

    close_kernel = np.ones((5, 5), np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, close_kernel)

    white = mask_out_detections(white, detections)

    edges = auto_canny(white)
    return edges


def filter_lines_by_geometry(lines, min_length=35, min_angle_deg=15, max_angle_deg=165):

    if lines is None:
        return None

    kept = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if angle < min_angle_deg or angle > max_angle_deg:
            continue
        kept.append(line)

    return np.array(kept) if kept else None


def _line_angle(line):
    x1, y1, x2, y2 = line[0]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def filter_lines_by_temporal_consistency(lines):

    global smoothed_angle_left, smoothed_angle_right

    if lines is None:
        return None

    center_x = IMAGE_WIDTH / 2
    left_group, right_group = [], []
    for line in lines:
        x1, _, x2, _ = line[0]
        mid_x = (x1 + x2) / 2
        (left_group if mid_x < center_x else right_group).append(line)

    def process_group(group, smoothed_angle):
        if not group:
            return [], smoothed_angle

        kept = []
        for line in group:
            angle = _line_angle(line)
            if smoothed_angle is None or abs(angle - smoothed_angle) <= ANGLE_TOLERANCE:
                kept.append(line)

        if kept:

            mean_angle = np.mean([_line_angle(l) for l in kept])
            smoothed_angle = (
                mean_angle if smoothed_angle is None
                else (1 - ANGLE_SMOOTHING) * smoothed_angle + ANGLE_SMOOTHING * mean_angle
            )
        return kept, smoothed_angle

    kept_left, smoothed_angle_left = process_group(left_group, smoothed_angle_left)
    kept_right, smoothed_angle_right = process_group(right_group, smoothed_angle_right)

    result = kept_left + kept_right
    return np.array(result) if result else None


def line_detection(frame, detections=None):
    global shared_lines, last_valid_lines, lost_frames_count

    edges = preprocess_for_lines(frame, detections)

    top_x, top_y = 0, 0
    end_x, end_y = IMAGE_WIDTH, IMAGE_HEIGHT // 2
    cv2.rectangle(edges, (top_x, top_y), (end_x, end_y), (0, 0, 0), -1)

    rho = 1
    theta = np.pi / 180
    threshold = 15
    minLineLength = 40
    maxLineGap = 30

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, None, minLineLength, maxLineGap)
    lines = filter_lines_by_geometry(lines)
    lines = filter_lines_by_temporal_consistency(lines)

    with lock:
        if lines is not None and len(lines) > 0:
            shared_lines = lines
            last_valid_lines = lines
            lost_frames_count = 0
        else:

            lost_frames_count += 1
            if last_valid_lines is not None and lost_frames_count <= MAX_LOST_FRAMES:
                shared_lines = last_valid_lines
            else:
                shared_lines = None


def yolo_inference(frame):
    global shared_detections
    results = model(frame, verbose=False, conf=0.7)

    with lock:
        shared_detections = results


def main():
    global IMAGE_HEIGHT, IMAGE_WIDTH
    video_source = source
    cap = cv2.VideoCapture(video_source)
    t_lines = None
    t_yolo = None

    if not cap.isOpened():
        print(f"Error: {video_source}")
        cap = cv2.VideoCapture(0)
    print("Press 'ESC' to exit or 'SPACE' to toggle pause/resume")

    ret, first_frame = cap.read()
    if not ret: return

    h, w = first_frame.shape[:2]
    final_h = h - crop_top - crop_bottom
    final_w = w - crop_left - crop_rigth
    scale = min(max_screen_width / final_w, max_screen_height / final_h, 1.0)

    IMAGE_WIDTH = int(final_w * scale)
    IMAGE_HEIGHT = int(final_h * scale)

    paused = False
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[crop_top:h - crop_bottom, crop_left:w - crop_rigth]
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # Create threading
            if t_lines is None or not t_lines.is_alive():
                with lock:
                    detections_for_mask = shared_detections
                t_lines = threading.Thread(
                    target=line_detection, args=(frame.copy(), detections_for_mask)
                )
                t_lines.start()

            if t_yolo is None or not t_yolo.is_alive():
                t_yolo = threading.Thread(target=yolo_inference, args=(frame.copy(),))
                t_yolo.start()

            with lock:
                current_lines = shared_lines
                current_detections = shared_detections

            if current_detections is not None:
                for r in current_detections:
                    annotated_frame = r.plot()
                    frame = annotated_frame

            if current_lines is not None:
                for line in current_lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Lane Follower", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Space to pause/resume
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()