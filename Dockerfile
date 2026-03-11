FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml uv.lock ./
RUN uv pip install --system .

RUN python -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8l-worldv2.pt')"

COPY . .

ENTRYPOINT ["python", "main.py"]