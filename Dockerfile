FROM nvcr.io/nvidia/pytorch:21.08-py3

ARG USER_ID
ARG GROUP_ID
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    tmux \
    python3-tk

RUN addgroup --gid $GROUP_ID docker 
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID docker
USER docker

ENV PATH="/home/docker/.local/bin:${PATH}"

RUN pip3 install kaggle \
                 pandas \
                 matplotlib \
                 numpy \
                 pytorch-lightning \
                 timm

WORKDIR /home/docker

