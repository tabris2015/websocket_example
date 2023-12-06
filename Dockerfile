FROM python:3.11-slim
ENV PORT 8000

RUN apt-get update && apt install wget ffmpeg libsm6 libxext6  -y
# install cpu version of pytorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY ./app /app
COPY ./assets /assets
COPY index.html index.html

ARG YOLO_VERSION
ENV YOLO_VERSION ${YOLO_VERSION}
RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/${YOLO_VERSION}

RUN wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}