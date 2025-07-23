# syntax=docker/dockerfile:1
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc g++ build-essential
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        --timeout 120 \
        --retries 5 \
        -r requirements.txt
COPY . .
ENTRYPOINT ["python", "-u", "train1.py"]