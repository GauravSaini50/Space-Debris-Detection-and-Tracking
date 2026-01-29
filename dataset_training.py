import argparse
from pathlib import Path
from kaggle import api
from ultralytics import YOLO

class DatasetManager:
    def __init__(self):
        self.root = Path('.').resolve()
        self.datasets_dir = self.root / 'datasets'
        self.datasets_dir.mkdir(exist_ok=True)

    def download_dataset(self, dataset_name: str):
        dest = self.datasets_dir / 'data'
        dest.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_name, path=str(dest), unzip=True, quiet=False)
        return dest

    def train_model(self, dataset_folder: str, model='yolov8n.pt', epochs=50, batch=16, device='0'):
        data_yaml = Path(f"datasets/{dataset_folder}/data.yaml")
        if not data_yaml.exists():
            raise FileNotFoundError(f"{data_yaml} not found")

        model_obj = YOLO(model)
        model_obj.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            device=device,
            imgsz=640,
            save=True,
            workers=8,
            cache=True
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', type=str)
    parser.add_argument('--train', type=str)
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    manager = DatasetManager()

    if args.download:
        manager.download_dataset(args.download)

    if args.train:
        manager.train_model(
            args.train,
            model=args.model,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device
        )

if __name__ == '__main__':
    main()
