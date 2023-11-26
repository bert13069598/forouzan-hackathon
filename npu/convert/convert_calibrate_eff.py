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
import furiosa.runtime.session


skip_convert = False

random.seed(0)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)), # H, W

])
# data_path_list = glob("/home/elicer/cat_dog_data/training_set/training_set/**/*.jpg")
data_path_list = glob("/home/elicer/dataset/eff_2/**/*.jpg")
print(len(data_path_list))
# data_path_list = random.sample(data_path_list, 200)


# eff
model_name = "efficientnet_b0"
model_weight_path = "/home/elicer/furiosa-hackathon/model/weight/eff_net1.pt"

torch_model = models.efficientnet_b0()
torch_model.classifier = nn.Linear(1280, 2)
torch_model.load_state_dict(torch.load(model_weight_path, map_location="cpu"))
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

    # with open(f"npu/convert_result/{model_name}_graph.pkl", "wb") as f:
    #     pkl.dump(graph, f)
    # onnx.save(graph, f"npu/convert_result/{model_name}_graph.onnx")
    # graph = onnx.load_model(f"npu/convert_result/{model_name}_graph.onnx")
    with open(f"npu/convert_result/{model_name}_graph.dfg", "wb") as f:
        f.write(bytes(graph))
    

else:
    with open(f"npu/convert_result/{model_name}_ranges.json") as f:
        ranges = json.load(f)

    # with open(f"npu/convert_result/{model_name}_graph.pkl", "rb") as f:
    #     graph = pkl.load(f)
    with open(f"npu/convert_result/{model_name}_graph.dfg", "rb") as f:
        graph = f.read()


print("evaluating")

from furiosa.runtime.sync import create_runner

# async 
# from furiosa.runtime import create_runner


acc_n = Accuracy(task="multiclass", num_classes=2)
# f1_n = F1Score(task="multiclass", num_classes=2)
acc_t = Accuracy(task="multiclass", num_classes=2)
# f1_t = F1Score(task="multiclass", num_classes=2)
# correct_n = 0
# correct_t = 0
# warboy(1)*1 warboy(1)*1
# npu10pe0, npu10pe1, npu10pe0-1
with furiosa.runtime.session.create(graph) as session:
# sess = furiosa.runtime.sync.create_runner('mnist-8.onnx', device="npu0pe0")
# with create_runner(graph, device="warboy(1)*1") as session:
    # for i in range(20):
    for data_path in tqdm(data_path_list):
        # data_path = data_path_list[i]
        target = int(data_path.split('/')[-2])
        # print(data_path)

        img = cv2.imread(data_path)
        img_t = transform(img).unsqueeze(0)

        t0 = time.time()
        out_n = session.run(img_t.numpy())[0].numpy()
        t1 = time.time()
        with torch.no_grad():
            out_t = torch_model(img_t).numpy()
        t2 = time.time()

        # print(t1 - t0, t2 - t1)
        # print(np.argmax(out_n[0]), np.argmax(out_t[0]))
        # print(target, np.argmax(out_n[0]), np.argmax(out_t[0]))
        # if target == np.argmax(out_n[0]):
        #     correct_n += 1
        # if target == np.argmax(out_t[0]):
        #     correct_t += 1

        conf = float(torch.softmax(torch.tensor(out_n), dim=1)[0,0].numpy())
        # print(conf)
        # print(f"{conf * 100:.1f}%")

        acc_n(tensor(out_n), tensor([target]))
        acc_t(tensor(out_t), tensor([target]))
        # f1_n(tensor(out_n), tensor([target]))
        # f1_t(tensor(out_t), tensor([target]))
    # print(correct_n / len(data_path_list), correct_t / len(data_path_list))
    # print(acc_n.compute(), f1_n.compute())
    # print(acc_t.compute(), f1_t.compute())
    print(acc_n.compute(), acc_t.compute())