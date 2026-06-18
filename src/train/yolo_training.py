from ultralytics import YOLO

model = YOLO('runs/detect/circuit_object-4/weights/best.pt')
result = model.train(
    data = "data/dataset/data.yaml",
    epochs = 50,
    batch = 16,
    name = "circuit_object",
    save_period = 5  # Save a checkpoint every 5 epochs
)


