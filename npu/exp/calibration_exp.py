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
import torchvision.models as models
from torchvision.transforms import v2
from tqdm import tqdm
from torchmetrics.classification import Accuracy, F1Score

from furiosa.optimizer import optimize_model
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
model_weight_path = "/home/elicer/furiosa-hackathon/model/weight/eff_net1.pt"

torch_model = models.efficientnet_b0()
torch_model.classifier = nn.Linear(1280, 2)
torch_model.load_state_dict(torch.load(model_weight_path, map_location="cpu"))
torch_model.eval()


if not skip_convert:
    onnx_model = onnx.load_model(f"npu/convert_result/{model_name}.onnx")
    onnx_model = optimize_model(onnx_model)

    calibrator = Calibrator(onnx_model, CalibrationMethod.MIN_MAX_ASYM)

    print("calibrating")

    for data_path in tqdm(random.sample(data_path_list, 200)[:200]):
        img = cv2.imread(data_path)
        img_t = transform(img).unsqueeze(0)
        calibrator.collect_data([[img_t.numpy()]])


    t0 = time.time()
    ranges = calibrator.compute_range()
    graph = quantize(onnx_model, ranges)
    t1 = time.time()
    print("time", t1 - t0)

    with open(f"npu/convert_result/{model_name}_{cal_method}_ranges.json", "w") as f:
        f.write(json.dumps(ranges, indent=4))

    # with open(f"npu/convert_result/{model_name}_{cal_method}_graph.pkl", "wb") as f:
    #     pkl.dump(graph, f)
    # onnx.save(graph, f"npu/convert_result/{model_name}_{cal_method}_graph.onnx")
    with open(f"npu/convert_result/{model_name}_{cal_method}_graph.dfg", "wb") as f:
        f.write(bytes(graph))

else:
    with open(f"npu/convert_result/{model_name}_{cal_method}_ranges.json") as f:
        ranges = json.load(f)

    # with open(f"npu/convert_result/{model_name}_{cal_method}_graph.pkl", "rb") as f:
    #     graph = pkl.load(f)
    # graph = onnx.load_model(f"npu/convert_result/{model_name}_{cal_method}_graph.onnx")
    with open(f"npu/convert_result/{model_name}_{cal_method}_graph.dfg", "rb") as f:
        graph = f.read()

print("evaluating")

# async 
# from furiosa.runtime import create_runner


acc_n = Accuracy(task="multiclass", num_classes=2)
acc_t = Accuracy(task="multiclass", num_classes=2)

lat_n, lat_t = 0, 0

# warboy(1)*1 warboy(2)*1
# npu10pe0, npu10pe1, npu10pe0-1
# with furiosa.runtime.session.create(graph) as session:
# with furiosa.runtime.session.create(graph, "npu10pe0") as session: 
# runner = furiosa.runtime.sync.create_runner('mnist-8.onnx', device="npu0pe0")
# with create_runner(graph, device="warboy(1)*1") as runner:
# with create_runner(f"npu/convert_result/{model_name}_{cal_method}_graph.dfg", device="npu10pe1") as runner:
with create_runner(graph, device="npu10pe1") as runner:
    for data_path in tqdm(data_path_list):
        target = int(data_path.split('/')[-2])

        img = cv2.imread(data_path)
        img_t = transform(img).unsqueeze(0)

        t0 = time.time()
        out_n = runner.run(img_t.numpy())[0]
        t1 = time.time()
        # with torch.no_grad():
        #     out_t = torch_model(img_t).numpy()
        t2 = time.time()

        acc_n(tensor(out_n), tensor([target]))
        # acc_t(tensor(out_t), tensor([target]))
        lat_n += t1 - t0
        lat_t += t2 - t1

    # print(acc_n.compute(), acc_t.compute())
    print(acc_n.compute())
    print(lat_n / len(data_path_list), lat_t / len(data_path_list))