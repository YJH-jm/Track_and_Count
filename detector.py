from dataclasses import dataclass

from ultralytics import YOLO

model_path = "./saved/yolov8x.pt"
model = YOLO(model_path)
model.fuse()