# Docker Testing Setup for TensorLight

This directory contains Docker configuration for testing TensorLight in a controlled environment.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for more advanced usage)

## Running Tests in Docker

You can use the provided script to build and run tests in Docker:

```bash
./run-tests-docker.sh
```

## Manual Docker Commands

If you prefer to run Docker commands manually:

### Build the Docker image

```bash
docker build -t tensorlight-test .
```

### Run tests in the container

```bash
docker run --rm tensorlight-test
```

### Using Docker Compose

```bash
docker-compose up --build
```

## Docker Configuration

- `Dockerfile`: Defines the Ubuntu 22.04 environment with all necessary build tools and dependencies
- `docker-compose.yml`: Simplifies Docker usage with volume mounting for development
- `.dockerignore`: Excludes unnecessary files from the Docker build context
- `run-tests-docker.sh`: Convenience script for building and running tests

## Continuous Integration

This Docker setup can be integrated with CI/CD pipelines to ensure consistent testing across environments.