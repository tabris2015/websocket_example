import asyncio
import io

import numpy as np
from PIL import Image
from fastapi import HTTPException, status

from starlette.websockets import WebSocket

from app.detector import ObjectDetector


async def receive(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes_data = await websocket.receive_bytes()
        try:
            queue.put_nowait(bytes_data)
        except asyncio.QueueFull:
            pass


async def detect(detector: ObjectDetector, websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes_data = await queue.get()
        image = Image.open(io.BytesIO(bytes_data))
        objects = detector.predict_image(np.array(image), 0.4)
        await websocket.send_json(objects.dict())


def predict_uploadfile(predictor, file, threshold):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Not an image"
        )
    # convertir a una imagen de Pillow
    img_obj = Image.open(img_stream)
    # crear array de numpy
    img_array = np.array(img_obj)
    return predictor.predict_image(img_array, threshold), img_array
