#!/bin/bash

docker build -t pytorch \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .
