from ultralytics import YOLO
import sys
import os

def predict_image(model_path, image_path):
    if not os.path.exists(model_path):
        print(f"The model '{model_path}'doesn't exist")
        return
    if not os.path.exists(image_path):
        print(f"The image '{image_path}' doesn't exist")
        return

    model = YOLO(model_path)

    results = model.predict(source=image_path, save=True, conf=0.25)

    for result in results:
        save_path = os.path.join(result.save_dir, os.path.basename(image_path))
        print(f"Result save in: {save_path}")
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            conf = float(box.conf[0])
            print(f" - {label} ({conf:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Do: python yolo.py <path_to_model.pt> <path_to_image.jpg>")
    else:
        predict_image(sys.argv[1], sys.argv[2])
