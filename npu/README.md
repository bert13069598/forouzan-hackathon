
## convert
변환 관련 

- convert_calibrate_eff.py: efficientnet calibration 및 변환
- convert_calibrate_fcn.py: fcn calibration 및 변환
- convert_exp_yolo.py: yolo 변환 모델 테스트
- convert_exp.py: efficientnet 변환 모델 테스트

## convert_result
변환이 완료된 json, onnx, dfg 파일들 보관하는 곳

## exp
- async_eff.py: efficientnet async 프로파일링
- async_yolo.py: yolo async 프로파일링
- calibration_exp.py: efficientnet calibration + 평가
- sync_fcn.py: fcn sync 프로파일링
- sync_yolo.py: yolo sync 프로파일링

## exp_out
실험 결과(프로파일링) json 파일 보관하는 곳

## inference
- yolo.py: yolo 추론 write