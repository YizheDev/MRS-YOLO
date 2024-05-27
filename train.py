from ultralytics import YOLO

model = YOLO("D:\yolov8-DSNV2.yaml")
result = model.train(data="data.yaml", epochs=5)