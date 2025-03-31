#!/bin/bash

# Script to build and run tests in Docker container

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the Docker image
docker build -t tensorlight-test -f docker/Dockerfile .

# Run the tests
docker run --rm tensorlight-test

# Alternatively, use docker-compose:
# cd docker && docker-compose up --build