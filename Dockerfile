FROM nvidia/cuda:11.8.0-devel-ubuntu20.04


WORKDIR /app

#installing python
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y --no-install-recommends python3.10 python3-pip
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

#installing libraries
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt