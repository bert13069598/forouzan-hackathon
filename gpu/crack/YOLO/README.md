



### convert format
```bash
python converter/converter.py -l crack_1
```

### train
```bash
python main.py
```


### onnx
```bash
python export.py \
--weights /home/elicer/hackathon/gpu/crack/YOLO/weights/yolov8/box/1126/weights/last.pt \
--onnx_path ./yolo_best.onnx
```


### quantized
```bash
python quantizer.py \
--onnx_path yolo_best.onnx \
--calib_data /home/elicer/dataset/crack_1/train/images
```
