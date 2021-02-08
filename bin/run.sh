#!/usr/bin/env bash

docker build -t melvin_image -f dockerfile_run .
docker run -it --rm melvin_image "$@"
