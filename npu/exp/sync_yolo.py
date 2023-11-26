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


skip_convert = True
cal_method = "minmax"

random.seed(0)

data_path_list = glob("/home/elicer/dataset/eff_2/**/*.jpg")
print(len(data_path_list))


# async 
# from furiosa.runtime import create_runner


# warboy(1)*1 warboy(2)*1
# npu10pe0, npu10pe1, npu10pe0-1
# with furiosa.runtime.session.create(graph) as session:
# with furiosa.runtime.session.create(graph, "npu10pe0") as session: 
# runner = furiosa.runtime.sync.create_runner('mnist-8.onnx', device="npu0pe0")
onnx_path = "/home/elicer/furiosa-hackathon/npu/crack/quantized_model.onnx"
json_path = "/home/elicer/furiosa-hackathon/npu/exp_out/async_yolo.json"

# with open(json_path, "w") as output:
#     with profile(file=output) as profiler:
#         with create_runner(onnx_path, device="npu10pe0") as runner:
#             for i, data_path in enumerate(tqdm(data_path_list[:20])):
    
#                 with profiler.record("preprocess") as record:
#                     input_img = cv2.imread(data_path)
#                     input_, preproc_params = preproc(input_img)
                
#                 with profiler.record("inference") as record:
#                     output = runner.run(input_)

#                 with profiler.record("postprocess") as record:
#                     predictions = postproc(output, 0.65, 0.35)

#                     assert len(predictions) == 1, f"{len(predictions)=}"

#                     predictions = predictions[0]

#                     num_predictions = predictions.shape[0]
#                     if num_predictions != 0:
#                         # cv2.imwrite(output_img_path, input_img)
#                         # return

#                         bboxed_img = draw_bbox(input_img, predictions, preproc_params)



runner = create_runner(onnx_path, device="npu10pe0")

def process():
    idx = random.randint(0, len(data_path_list) - 1)
    input_img = cv2.imread(data_path_list[idx])
    input_, preproc_params = preproc(input_img)

    output = runner.run(input_)

    # predictions = postproc(output, 0.65, 0.35)
    # assert len(predictions) == 1, f"{len(predictions)=}"

    # predictions = predictions[0]

    # num_predictions = predictions.shape[0]
    # if num_predictions != 0:
    #     # cv2.imwrite(output_img_path, input_img)
    #     # return

    #     bboxed_img = draw_bbox(input_img, predictions, preproc_params)


for i in range(50):
    process()
t1 = time.time()
for i in range(100):
    process()
t2 = time.time()
print("time", (t2 - t1))