FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Python packages (must match AVAILABLE_LIBRARIES in config files)
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-image \
    scikit-learn \
    pandas \
    matplotlib \
    bioio \
    bioio-tifffile

WORKDIR /work
