FROM python:3.11-slim
ENV PORT 8000

RUN apt-get update && apt install wget ffmpeg libsm6 libxext6  -y
# install cpu version of pytorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /
RUN pip install -r requirements.txt
RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

COPY ./app /app
COPY ./assets /assets
COPY index.html index.html


CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}