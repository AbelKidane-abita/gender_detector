FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

#installing python
# RUN apt-get -y update && apt -get install software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa && apt install python3.10

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.10 \
#     python3-pip \
#     && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# compile python from source - avoid unsupported library problems
RUN apt update -y && sudo apt upgrade -y && \
    apt-get install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /usr/src && \
    sudo wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    sudo tar xzf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    sudo ./configure --enable-optimizations && \
    sudo make altinstall