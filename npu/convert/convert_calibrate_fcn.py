from glob import glob
import json
import random
import pickle as pkl
import time

import cv2
import numpy as np
import onnx
import torch
from torch import tensor
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights, fcn_resnet101, FCN_ResNet101_Weights
import torchvision.models as models
from torchvision.transforms import v2
from tqdm import tqdm
from torchmetrics.classification import Accuracy, F1Score

from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
import furiosa.runtime.session


skip_convert = False

random.seed(0)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)), # H, W

])
data_path_list = glob("/home/elicer/cat_dog_data/training_set/training_set/**/*.jpg")
# data_path_list = glob("/home/elicer/dataset/eff_2/**/*.jpg")
print(len(data_path_list))
data_path_list = random.sample(data_path_list, 200)


# eff
model_name = "fcn101"
# model_weight_path = "/home/elicer/furiosa-hackathon/model/weight/eff_net1.pt"

weights = FCN_ResNet101_Weights.DEFAULT
torch_model = fcn_resnet101(weights=weights)
torch_model.eval()

if not skip_convert:
    torch.onnx.export(
        torch_model,
        torch.randn((1, 3, 224, 224)),
        f"npu/convert_result/{model_name}.onnx",
        opset_version=13,  
        do_constant_folding=True, 
        input_names=["input"], 
        output_names=["output"], 
    )

    onnx_model = onnx.load_model(f"npu/convert_result/{model_name}.onnx")
    onnx_model = optimize_model(onnx_model)

    calibrator = Calibrator(onnx_model, CalibrationMethod.MIN_MAX_ASYM)
    

    print("calibrating")

    for data_path in tqdm(random.sample(data_path_list, 200)):
        img = cv2.imread(data_path)
        img_t = transform(img).unsqueeze(0)
        calibrator.collect_data([[img_t.numpy()]])

        ranges = calibrator.compute_range()

    graph = quantize(onnx_model, ranges)

    with open(f"npu/convert_result/{model_name}_ranges.json", "w") as f:
        f.write(json.dumps(ranges, indent=4))

    with open(f"npu/convert_result/{model_name}_graph.dfg", "wb") as f:
        f.write(bytes(graph))
    
else:
    with open(f"npu/convert_result/{model_name}_ranges.json") as f:
        ranges = json.load(f)

    with open(f"npu/convert_result/{model_name}_graph.dfg", "rb") as f:
        graph = f.read()


print("evaluating")

from furiosa.runtime.sync import create_runner


# warboy(1)*1 warboy(1)*1
# npu10pe0, npu10pe1, npu10pe0-1
# with furiosa.runtime.session.create(graph) as session:
# sess = furiosa.runtime.sync.create_runner('mnist-8.onnx', device="npu0pe0")

tt1 = 0
tt2 = 0

with create_runner(graph, device="npu10pe0") as session:
    for i in range(20):
    # for data_path in tqdm(data_path_list):
        data_path = data_path_list[i]

        img = cv2.imread(data_path)
        img_t = transform(img).unsqueeze(0)

        t0 = time.time()
        out_n = session.run(img_t.numpy())[0]
        t1 = time.time()
        with torch.no_grad():
            out_t = torch_model(img_t)["out"]
        t2 = time.time()
        out_t = out_t.softmax(dim=1).numpy()

        # print(out_n.shape, out_t.shape)

        # print(t1 - t0, t2 - t1)
        tt1 += t1 - t0
        tt2 += t2 - t1

print(tt1 / 20, tt2 / 20)