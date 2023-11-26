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

from npu.crack.utils.preprocess import preproc
from npu.crack.utils.postprocess import postproc, draw_bbox
# from npu.crack.inference import predict

from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
from furiosa.runtime.sync import create_runner
from furiosa.runtime.profiler import profile


skip_convert = False
cal_method = "minmax"

random.seed(0)

data_path_list = glob("/home/elicer/dataset/eff_2/**/*.jpg")
# data_path_list = glob("/home/elicer/dataset/bmp/*.bmp")
print(len(data_path_list))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)), # H, W

])


# warboy(1)*1 warboy(2)*1
# npu10pe0, npu10pe1, npu10pe0-1

onnx_path = "/home/elicer/furiosa-hackathon/npu/convert_result/fcn101_graph.dfg"
# json_path = "/home/elicer/furiosa-hackathon/npu/exp_out/sync_fcn101.json"

# with open(json_path, "w") as output:
#     with profile(file=output) as profiler:
#         with create_runner(onnx_path, device="npu10pe0") as runner:
#             for i, data_path in enumerate(tqdm(data_path_list[:20])):
    
#                 with profiler.record("read") as record:
#                     input_img = cv2.imread(data_path)
#                 with profiler.record("trasnform") as record:
#                     in_t = transform(input_img)[None]
                
#                 with profiler.record("inference") as record:
#                     output = runner.run(in_t)


runner = create_runner(onnx_path, device="npu10pe0")

def process():
    idx = random.randint(0, len(data_path_list) - 1)
    input_img = cv2.imread(data_path_list[idx])
    input_, preproc_params = preproc(input_img)

    output = runner.run(input_)



for i in range(50):
    process()
t1 = time.time()
for i in range(100):
    process()
t2 = time.time()
print("time", (t2 - t1))