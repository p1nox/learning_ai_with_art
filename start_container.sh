#!/usr/bin/env bash

IMAGE_NAME="learning_ai_with_art"

docker run -it -v $(pwd)/assets/inputs:/home/code/assets/inputs -v $(pwd)/assets/outputs:/home/code/assets/outputs ${IMAGE_NAME} bash
