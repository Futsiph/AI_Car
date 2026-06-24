import cv2
import numpy as np
import threading
from ultralytics import YOLO

model = YOLO("../models/best.pt")
model.to('cuda')
source = "../data/Video Πίστας Κτίριο Ζ 1ος όροφος - Περικλής Ντάφος/2025-01-08 10-25-09.mkv"
IMAGE_HEIGHT, IMAGE_WIDTH = 0, 0

#Crop if needed
crop_top = 120
crop_bottom = 50
crop_left = 155
crop_rigth = 170

max_screen_width = 1024
max_screen_height = 576

# Share var
shared_lines = None
shared_detections = None
lock = threading.Lock()

def line_detection(frame):

    global shared_lines
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    minVal = 100
    maxVal = 200
    edges = cv2.Canny(gray, minVal, maxVal)

    top_x, top_y = 0, 0
    end_x, end_y = IMAGE_WIDTH, IMAGE_HEIGHT // 2
    cv2.rectangle(edges, (top_x, top_y), (end_x, end_y), (0, 0, 0), -1)
    
    rho = 1
    theta = np.pi / 180
    threshold = 5
    minLineLength = 40
    maxLineGap = 60
    
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, None, minLineLength, maxLineGap)
    
    with lock:
        shared_lines = lines

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
    print("Press 'ESC' to exit")

    ret, first_frame = cap.read()
    if not ret: return

    h, w = first_frame.shape[:2]
    final_h = h - crop_top - crop_bottom
    final_w = w - crop_left - crop_rigth
    scale = min(max_screen_width / final_w, max_screen_height / final_h,1.0)

    IMAGE_WIDTH = int(final_w * scale)
    IMAGE_HEIGHT = int(final_h * scale)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[crop_top:h-crop_bottom, crop_left:w-crop_rigth]
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Create threading
        if t_lines is None or not t_lines.is_alive():
            t_lines = threading.Thread(target=line_detection, args=(frame.copy(),))
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

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
