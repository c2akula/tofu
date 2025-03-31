FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV CC=gcc
ENV CXX=g++

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    pkg-config \
    check \
    python3 \
    python3-pip \
    python3-numpy \
    git \
    # Add any other dependencies here
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a workspace directory
WORKDIR /app

# Copy the TensorLight source code
COPY . .

# Clean any existing build artifacts, configure and build the library
RUN make clean && ./configure && make lib

# Add a health check to ensure tests can run
HEALTHCHECK --interval=5s --timeout=5s --retries=3 CMD [ -f "build/test/test_tensorlight" ] || exit 1

# Run the tests
CMD ["make", "test"]

# Label the image
LABEL maintainer="TensorLight Maintainer" \
      version="0.1.0" \
      description="Docker image for testing TensorLight library"