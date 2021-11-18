#!/bin/bash

IMAGE_NAME=pytorch:latest

docker run --gpus all -it --net=host --ipc=host --rm -e DISPLAY=$DISPLAY \
                                                -v /tmp/.X11-unix:/tmp/.X11-unix \
                                                -v $XAUTHORITY:/root/.Xauthority \
                                                -v $PWD:/home/docker/pawpularity \
                                                --user $(id -u):$(id -g) \
                                                $IMAGE_NAME

