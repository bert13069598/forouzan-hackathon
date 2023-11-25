import argparse

import timm
import torch
import torchvision

from ultralytics import YOLO


def argparser():
    parser = argparse.ArgumentParser('EXPORT ONNX')
    parser.add_argument('--module', type=str, default='yolo', help='yolo hub vision timm')
    parser.add_argument('--model', type=str, default='yolo')
    parser.add_argument('-c', '--checkpoint', type=str, default='yolov8s')
    parser.add_argument('--device', type=str, default=0)
    return parser.parse_args()


if torch.cuda.is_available():
    DEVICE = torch.device('cuda:{}'.format((lambda args: args.device)(argparser())))
    print(f'Device: {DEVICE}')
    print(torch.cuda.get_device_name(DEVICE))
else:
    DEVICE = torch.device('cpu')
    print(f'Device: {DEVICE}')

def yolo(args):
    model = YOLO(f'onnx/{args.checkpoint}.pt')
    model.export(format="onnx", opset=13)

def hub(args):
    if args.model == 'ssd':
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                               model='nvidia_ssd',
                               weights='DEFAULT',
                               trust_repo=True)
        inputs = torch.randn(1, 3, 960, 540)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0',
                               model=args.checkpoint,
                               weights='DEFAULT')
        inputs = torch.randn(1, 3, 960, 540)
    # torch.save(model, f'onnx/{args.checkpoint}.pth')
    # model = torch.load(f'onnx/{args.checkpoint}.pth', map_location=DEVICE)

    torch.onnx.export(
        model,  # model being runtorch
        inputs,  # model input (or a tuple for multiple inputs)
        f"onnx/{args.checkpoint}.onnx",  # where to save the model (can be a file or file-like   object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['inputs'],  # the model's input names
        output_names=['outputs'])  # the model's output names


def vision(args):
    weights_map = {
                   'efficientnet_b0': 'EfficientNet_B0_Weights',
                   }
    model = getattr(torchvision.models, args.checkpoint)(weights=f'{weights_map[args.checkpoint]}.DEFAULT').to(DEVICE)
    inputs = torch.randn(1, 3, 224, 224, device=DEVICE)
    # torch.save(model, f'onnx/{args.checkpoint}.pth')
    # model = torch.load(f'onnx/{args.checkpoint}.pth', map_location=DEVICE)

    torch.onnx.export(
        model,
        inputs,
        f"onnx/{args.checkpoint}.onnx",
        opset_version=13,
        input_names=['inputs'],
        output_names=["outputs"])


def vision_od(args):
    weights_map = {
                   'maskrcnn_resnet50_fpn': 'MaskRCNN_ResNet50_FPN_Weights',
                   'maskrcnn_resnet50_fpn_v2': 'MaskRCNN_ResNet50_FPN_V2_Weights',
                   }
    model = getattr(torchvision.models.detection, args.checkpoint)(
        weights=f'{weights_map[args.checkpoint]}.COCO_V1').to(DEVICE)
    inputs = torch.randn(1, 3, 224, 224, device=DEVICE)
    # torch.save(model, f'onnx/{args.checkpoint}.pth')
    # model = torch.load(f'onnx/{args.checkpoint}.pth', map_location=DEVICE)

    torch.onnx.export(
        model,
        inputs,
        f"onnx/{args.checkpoint}.onnx",
        opset_version=13,
        input_names=['inputs'],
        output_names=["outputs"])


def main():
    args = argparser()
    print('model', args.model)
    print('checkpoint', args.checkpoint)

    eval(args.module)(args)


if __name__ == '__main__':
    print('EXPORT ONNX')
    main()
