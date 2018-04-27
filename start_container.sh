#!/usr/bin/env bash

IMAGE_NAME="learning_ai_with_art"

docker run -it -v $(pwd):/home/code ${IMAGE_NAME} bash
