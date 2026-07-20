from ultralytics import YOLO

def train():
    model = YOLO('models/yolov8n.pt')
    model.to('cuda')
    result = model.train(
        data = "data/dataset.yaml",
        epochs = 200,
        batch = 16,
        patience = 5,
        name = "circuit_object",
        save_period = 5,
        mosaic = True,
        mixup = 0.3,
        lr0=0.001,
        workers = 4
    )

if __name__ == '__main__':
    train()


