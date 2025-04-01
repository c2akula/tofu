#!/bin/bash

# Script to build and run tests in Docker container

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the Docker image
docker build -t tofu-test -f docker/Dockerfile .

# Run the tests
docker run --rm tofu-test

# Alternatively, use docker-compose:
# cd docker && docker-compose up --build