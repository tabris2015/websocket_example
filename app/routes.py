import asyncio
import time
from collections import Counter
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Depends, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from app.db import get_session
from app.detector import ObjectDetector, MediapipeObjectDetector
from app.models import Detection, DetectionEvent
from app.utils import predict_uploadfile, receive, detect

router = APIRouter()


def get_detector():
    return MediapipeObjectDetector()


@router.post("/objects")
def detect_objects(
        threshold: float = 0.5,
        file: UploadFile = File(...),
        predictor: ObjectDetector = Depends(get_detector),
        db_session: Session = Depends(get_session)
) -> Detection:
    start_time = time.time()
    results, _ = predict_uploadfile(predictor, file, threshold)
    inference_time = time.time() - start_time
    label_counts = Counter(results.labels)
    top_label = max(label_counts, key=label_counts.get)

    detection_event = DetectionEvent(
        pred_type=results.pred_type,
        n_detections=results.n_detections,
        detection_model=predictor.__class__.__name__,
        inference_time_ms=inference_time * 1000,
        top_detection_label=top_label,
        top_detection_count=label_counts[top_label],
        timestamp=datetime.now()
    )
    db_session.add(detection_event)
    db_session.commit()
    db_session.refresh(detection_event)

    return results


@router.get("/events")
def get_detection_events(db_session: Session = Depends(get_session)):
    events = db_session.exec(select(DetectionEvent)).all()
    return events


@router.websocket("/object-detection")
async def ws_object_detection(websocket: WebSocket, detector: ObjectDetector = Depends(get_detector)):
    await websocket.accept()

    queue = asyncio.Queue(maxsize=1)
    receive_task = asyncio.create_task(receive(websocket, queue))
    detect_task = asyncio.create_task(detect(detector, websocket, queue))

    try:
        done, pending = await asyncio.wait(
            {receive_task, detect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
