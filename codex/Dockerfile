FROM ubuntu:24.04

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y curl ca-certificates git \
    && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y python3.12 python-is-python3 \
    && apt-get update && apt-get install -y libgl1 ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y nodejs \
    && npm install -g @openai/codex \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*