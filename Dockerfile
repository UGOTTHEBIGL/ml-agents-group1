FROM --platform=linux/amd64 python:3.10.12-slim

# libraries the Unity game needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa libxcursor1 libxrandr2 libxi6 && rm -rf /var/lib/apt/lists/*

# fixed versions
RUN pip install "pip==24.2" "setuptools<81" && \
    pip install "torch==2.8.0+cpu" --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple

COPY ml-agents-envs /tmp/ml-agents-envs
COPY ml-agents /tmp/ml-agents
RUN pip install /tmp/ml-agents-envs /tmp/ml-agents && \
    pip install psutil "protobuf==3.20.1" "grpcio==1.48.2" "numpy==1.23.5"

WORKDIR /workspace
