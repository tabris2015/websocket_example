from datetime import datetime

from pydantic import BaseModel
from sqlmodel import SQLModel, Field
from enum import Enum


class PredictionType(str, Enum):
    classification = "CLS"
    object_detection = "OD"
    segmentation = "SEG"


class GeneralPrediction(BaseModel):
    pred_type: PredictionType


class Detection(GeneralPrediction):
    n_detections: int
    boxes: list[list[int]]
    labels: list[str]
    confidences: list[float]


class DetectionEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    pred_type: PredictionType
    detection_model: str
    architecture: str | None = Field(default=None)
    inference_time_ms: float
    n_detections: int
    top_detection_label: str
    top_detection_count: int
    timestamp: datetime
