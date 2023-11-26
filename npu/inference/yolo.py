import json
import os
import pickle as pkl
import time
from glob import glob
import random
import pathlib

import cv2
import numpy as np
import onnx
import torch
import torchvision
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm

from npu.crack.utils.preprocess import preproc
from npu.crack.utils.postprocess import postproc, draw_bbox


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)), # H, W

])

with open(f"npu/crack/quantized_model.onnx", "rb") as f:
    graph = f.read()

data_path_list = glob("/home/elicer/dataset/eff_2/**/*.jpg")
print(len(data_path_list))

from furiosa.runtime.sync import create_runner
with create_runner(graph, device="npu10pe0") as runner:
    for i in range(5):
        data_path = data_path_list[i]
        img = cv2.imread(data_path)
        img_t = transform(img).unsqueeze(0)

        start = time.time()

        input_, preproc_params = preproc(img)
        output = runner.run([img_t])
        predictions = postproc(output, 0.65, 0.35)

        assert len(predictions) == 1, f"{len(predictions)=}"

        predictions = predictions[0]

        num_predictions = predictions.shape[0]
        # if num_predictions == 0:
        #     cv2.imwrite(output_img_path, img)
        # else:
        #     bboxed_img = draw_bbox(img, predictions, preproc_params)
        #     cv2.imwrite(output_img_path, bboxed_img)
        if num_predictions != 0:
            bboxed_img = draw_bbox(img, predictions, preproc_params)
        else:
            bboxed_img = img

        img_name = pathlib.Path(data_path).name
        cv2.imwrite(f"/home/elicer/tmp/{img_name}", bboxed_img)
