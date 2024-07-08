import numpy as np
from ultralytics import YOLO
import supervision as sv

class Detector:
    def __init__(self, 
                 model_path:str,
                 confidence_threshold: float=0.3,
                 iou_threshold: float=0.7,
                 fuse:bool = True):
    
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold =iou_threshold
        if fuse:
            self.model.fuse()

    def detection(self, frame: np.ndarray):
        result = self.model(frame, verbose=False, conf=self.confidence_threshold, iou=self.iou_threshold)[0]
        result = sv.Detections.from_ultralytics(result)
        return result 