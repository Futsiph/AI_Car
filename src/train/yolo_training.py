from ultralytics import YOLO

def train():
    model = YOLO('models/best.pt')
    model.to('cuda')
    result = model.train(
        data = "data/dataset/data.yaml",
        epochs = 50,
        batch = 16,
        name = "circuit_object",
        save_period = 5  # Save a checkpoint every 5 epochs
    )

if __name__ == '__main__':
    train()


