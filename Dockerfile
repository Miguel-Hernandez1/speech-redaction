FROM python:3.12-slim

# System deps for audio (soundfile needs libsndfile)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BIRDNET_YAMNET_TFLITE=/app/models/yamnet.tflite

ENTRYPOINT ["python3", "main.py"]
