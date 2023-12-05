import asyncio
import contextlib
import io

import numpy as np
from app.detector import ObjectDetector
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel


app = FastAPI()
detector = ObjectDetector()


async def receive(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await websocket.receive_bytes()
        try:
            queue.put_nowait(bytes)
        except asyncio.QueueFull:
            pass


async def detect(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await queue.get()
        image = Image.open(io.BytesIO(bytes))
        objects = detector.predict_image(np.array(image), 0.2)
        await websocket.send_json(objects.dict())


@app.websocket("/object-detection")
async def ws_object_detection(websocket: WebSocket):
    await websocket.accept()

    queue = asyncio.Queue(maxsize=1)
    receive_task = asyncio.create_task(receive(websocket, queue))
    detect_task = asyncio.create_task(detect(websocket, queue))

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