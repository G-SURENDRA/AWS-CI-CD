
FROM python:3.8-slim-bullseye

WORKDIR /app
COPY . /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        awscli \
        ffmpeg \
        libsm6 \
        libxext6 \
        unzip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt


CMD ["python", "app.py"]
