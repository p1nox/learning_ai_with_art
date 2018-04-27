#!/usr/bin/env bash

IMAGE_NAME="learning_ai_with_art"

echo "Building image ${IMAGE_NAME}"
docker build -t ${IMAGE_NAME} .

echo "All set!  Successfully built ${IMAGE_NAME}"
