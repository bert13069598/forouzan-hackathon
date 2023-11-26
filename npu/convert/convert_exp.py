import json
import pickle as pkl
import time

import numpy as np
import onnx
import torch
import torchvision
from torchvision import transforms
import torchvision.models as models
import tqdm

from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
import furiosa.runtime.session


# eff
model_name = "efficientnet_b0"
torch_model = models.efficientnet_b0(weights=models.efficientnet.EfficientNet_B0_Weights)



dummy_input = (torch.randn(1, 3, 224, 224),)
dummy_np = np.random.rand(1,3,224,224).astype(np.float32)


# Load the exported ONNX model.
onnx_model = onnx.load_model(f"npu/convert/{model_name}.onnx")

onnx_model = optimize_model(onnx_model)

calibrator = Calibrator(onnx_model, CalibrationMethod.MIN_MAX_ASYM)

for i in range(10):
    calibrator.collect_data([[dummy_np]])

    ranges = calibrator.compute_range()

graph = quantize(onnx_model, ranges)

with open(f"npu/convert/{model_name}_ranges.json", "w") as f:
    f.write(json.dumps(ranges, indent=4))

graph = quantize(onnx_model, ranges)
elapsed_time = 0

with open(f"npu/convert/{model_name}_graph.pkl", "wb") as f:
    pkl.dump(graph, f)

with furiosa.runtime.session.create(graph) as session:
    for i in range(10):
        image = dummy_np

        start = time.perf_counter_ns()
        outputs = session.run(image)
        elapsed_time += time.perf_counter_ns() - start
print(elapsed_time)