from ultralytics import YOLO

def train():
    model = YOLO('runs/detect/circuit_object-7/weights/best.pt')
    model.to('cuda')
    result = model.train(
        data = "data/dataset2/data.yaml",
        epochs = 50,
        batch = 16,
        name = "circuit_object",
        save_period = 5,
        workers = 4,
        exist_ok = True,
        pretrained = True
    )

if __name__ == '__main__':
    train()


