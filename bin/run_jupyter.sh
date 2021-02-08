#!/usr/bin/env bash

docker build -t melvin_jupyter -f dockerfile_jupyter .
docker container run --rm -p 8888:8888 -v $(pwd):/home/jovyan/ melvin_jupyter
