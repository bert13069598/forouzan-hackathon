### furiosa-hackathon
노후 건물 균열 정도를 AI로 예측

```
.
├── dataset
└── hackathon
    ├── README.md
    ├── gpu
    │   └── crack
    └── npu
        └── crack
```

데이터셋은 AIHub에 있는 서울시 노후 주택 균열 데이터 데이터 데이터셋을 이용

용량이 너무 커서 전체를 이용하지는 못하고 일부만 다운받아 학습 계획

gpu/crack의 efficientnet, maskrcnn, ssd를 통해 분류, segmentation, detection 감지를 위한

학습하려 했지만 efficientnet만 npu에서 구동 가능하여 분류만 진행.

detection은 yolov8을 이용하여 진행.
AIHub의 label format이 yolo형식이 아니어서 변경 후 진행