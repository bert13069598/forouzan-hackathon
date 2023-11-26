from glob import glob
import json
import random
import pickle as pkl
import time
import asyncio

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



onnx_path = "/home/elicer/furiosa-hackathon/npu/crack/quantized_model.onnx"
json_path = "/home/elicer/furiosa-hackathon/npu/exp_out/async_yolo.json"

runner = create_runner(onnx_path, device="npu10pe0")

async def process():
    idx = random.randint(0, len(data_path_list) - 1)
    input_img = cv2.imread(data_path_list[idx])
    input_, preproc_params = preproc(input_img)
    
    output = runner.run(input_)

    # predictions = postproc(output, 0.65, 0.35)

    # assert len(predictions) == 1, f"{len(predictions)=}"

    # predictions = predictions[0]

    # num_predictions = predictions.shape[0]
    # if num_predictions != 0:
    #     bboxed_img = draw_bbox(input_img, predictions, preproc_params)


async def run():
    await asyncio.gather(
        *(process() for i in range(50))
    )
    t1 = time.time()
    await asyncio.gather(
        *(process() for i in range(3000))
    )
    t2 = time.time()
    print("time", (t2 - t1))

loop = asyncio.get_event_loop()
loop.run_until_complete(run())
loop.close()