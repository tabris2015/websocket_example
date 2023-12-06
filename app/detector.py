import time
from typing import Protocol

import numpy as np
from ultralytics import YOLO
from app.config import get_settings
from app.models import Detection, PredictionType
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

SETTINGS = get_settings()

class ObjectDetector(Protocol):
    def predict_image(self, image_array: np.ndarray, threshold: float):
        pass


class YOLOObjectDetector:
    def __init__(self) -> None:
        self.model = YOLO(SETTINGS.yolo_version)

    def predict_image(self, image_array: np.ndarray, threshold: float):
        results = self.model(image_array, conf=threshold)[0]
        labels = [results.names[i] for i in results.boxes.cls.tolist()]
        boxes = [[int(v) for v in box] for box in results.boxes.xyxy.tolist()]
        detection = Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels,
            confidences=results.boxes.conf.tolist()
        )
        return detection


class MediapipeObjectDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=SETTINGS.mediapipe_det_model)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            max_results=10,
            score_threshold=0.5,
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.detection_result_list = []

    def save_result(self, result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        self.detection_result_list.append(result)

    def predict_image(self, img_array: np.ndarray, threshold: float):
        # convert image for tflite model
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        result = self.detector.detect(mp_image)
        boxes = []
        labels = []
        confidences = []
        for detection in result.detections:
            bbox = detection.bounding_box
            boxes.append([bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height])
            labels.append(detection.categories[0].category_name)
            confidences.append(detection.categories[0].score)
        detection = Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels,
            confidences=confidences
        )
        return detection
