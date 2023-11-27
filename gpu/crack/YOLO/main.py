import argparse
import os
from datetime import datetime

from ultralytics import YOLO


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='yolov8s')
    parser.add_argument('--device', type=str, default=0)
    return parser.parse_args()


def export_yolo(args):
    model = YOLO(f'weights/{args.checkpoint}.pt')
    model.export(format="onnx", opset=13)


def main():
    args = argparser()

    os.makedirs('weights', exist_ok=True)
    model = YOLO(f'weights/{args.checkpoint}.pt', task='detect')
    model.train(data='config/yolov8-box.yaml',
                epochs=20,
                batch=32,
                device=int(args.device),
                patience=30,
                project='weights/yolov8/box',
                name=f'{datetime.now().strftime("%m%d")}')
    export_yolo(args)


if __name__ == '__main__':
    main()