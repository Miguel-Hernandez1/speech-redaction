FROM waggle/plugin-base:1.1.1-base

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV BIRDNET_YAMNET_TFLITE=/app/models/yamnet.tflite

ENTRYPOINT ["python3", "main.py"]
