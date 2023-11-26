import json
import os
import pickle as pkl
import time
from glob import glob
import random

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

from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod

skip_convert = False


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)), # H, W

])

data_path_list = glob("/home/elicer/dataset/eff_2/**/*.jpg")
print(len(data_path_list))

if not skip_convert:
    onnx_model = onnx.load_model(f"npu/convert_result/yolo.onnx")
    onnx_model = optimize_model(onnx_model)
    calibrator = Calibrator(onnx_model, CalibrationMethod.MIN_MAX_ASYM)

    print("calibrating")

    for data_path in tqdm(random.sample(data_path_list, 200)):
        img = cv2.imread(data_path)
        # input_, preproc_params = preproc(img)
        img_t = transform(img).unsqueeze(0)
        # print(input_.max(), input_.min())
        # calibrator.collect_data([[input_]])
        calibrator.collect_data([[img_t.numpy()]])
    ranges = calibrator.compute_range()
    graph = quantize(onnx_model, ranges)

    with open(f"npu/convert/yolo_ranges.json", "w") as f:
        f.write(json.dumps(ranges, indent=4))

    with open(f"npu/convert_result/yolo_graph.dfg", "wb") as f:
            f.write(bytes(graph))
else:
    with open(f"npu/convert_result/yolo_ranges.json") as f:
        ranges = json.load(f)

    with open(f"npu/convert_result/yolo_graph.dfg", "rb") as f:
        graph = f.read()

from furiosa.runtime.sync import create_runner
with create_runner(graph, device="npu10pe0") as runner:
    for i in range(10):
        data_path = data_path_list[i]
        img = cv2.imread(data_path)
        img_t = transform(img).unsqueeze(0)

        start = time.time()

        input_, preproc_params = preproc(img)
        # output = predict(runner, input_)
        # output = runner.run([input_])
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



print(elapsed_time)