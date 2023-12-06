import asyncio

from app.detector import ObjectDetector, YOLOObjectDetector, MediapipeObjectDetector
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import Detection
from app.utils import receive, detect, predict_uploadfile

app = FastAPI()


def get_detector():
    return MediapipeObjectDetector()


@app.post("/objects")
def detect_objects(
        threshold: float = 0.5,
        file: UploadFile = File(...),
        predictor: ObjectDetector = Depends(get_detector)
) -> Detection:
    results, _ = predict_uploadfile(predictor, file, threshold)

    return results


@app.websocket("/object-detection")
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


@app.get("/")
async def index():
    return FileResponse("index.html")

static_files_app = StaticFiles(directory="assets")

app.mount("/assets", static_files_app)
