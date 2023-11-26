from glob import glob
import json
import random
import pickle as pkl
import time

import cv2
import numpy as np
from torch import tensor
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
from furiosa.runtime.sync import create_runner


skip_convert = True
cal_method = "minmax"

random.seed(0)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)), # H, W

])
data_path_list = glob("/home/elicer/dataset/eff_2/**/*.jpg")
print(len(data_path_list))


# eff
model_name = "efficientnet_b0"

with open(f"npu/convert_result/{model_name}_{cal_method}_ranges.json") as f:
    ranges = json.load(f)

with open(f"npu/convert_result/{model_name}_{cal_method}_graph.pkl", "rb") as f:
    graph = pkl.load(f)

print("evaluating")



# async 
# from furiosa.runtime import create_runner


# warboy(1)*1 warboy(2)*1
# npu10pe0, npu10pe1, npu10pe0-1
# with furiosa.runtime.session.create(graph) as session:
# with furiosa.runtime.session.create(graph, "npu10pe0") as session: 
# runner = furiosa.runtime.sync.create_runner('mnist-8.onnx', device="npu0pe0")
with create_runner(graph, device="warboy(1)*1") as runner:
    for i, data_path in enumerate(tqdm(data_path_list)):
        target = int(data_path.split('/')[-2])

        img = cv2.imread(data_path)
        img_t = transform(img).unsqueeze(0)

        t0 = time.time()
        out_n = runner.run(img_t.numpy())[0]
        t1 = time.time()

        t2 = time.time()

