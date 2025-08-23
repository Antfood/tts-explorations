FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# whisperx needs cudnn v8 and librosa ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcudnn8 \
        libcudnn8-dev \
        awscli \
        ffmpeg \
        ca-certificates curl git && \
        nvim \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r librosa boto3 rich whisperx 

WORKDIR /workspace
COPY . .

CMD ["bash"]
