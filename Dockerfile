FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget unzip \
    ffmpeg libsm6 libxext6 libjpeg-dev libpng-dev \
    python3 python3-pip python3-dev python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Create non-root user (match host UID/GID)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} devuser && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} devuser

USER devuser
WORKDIR /workspace
