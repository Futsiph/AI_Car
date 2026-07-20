import os
from pathlib import Path
import cv2


def extract_image(video_path, output_dir, frame_interval):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    video_name = Path(video_path).stem
    count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if count % frame_interval == 0:
            frame_name = f"{video_name}_frame{count}.jpg"
            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, frame)
        count += 1
    capture.release()
    print("End")


class Preprocess:
    def __init__(self, size=(200, 200)):
        self.size = size

    def preprocess_image(self, image):
        post_image = cv2.resize(image, self.size)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2GRAY)
        post_image = cv2.GaussianBlur(post_image, (3, 3), 0)
        return post_image

if __name__ == "__main__":
    videos = []

    for v_path, out_dir in videos:
        if os.path.exists(v_path):
            extract_image(v_path, out_dir, 10)
        else:
            print(f"no file found : {os.path.abspath(v_path)}")