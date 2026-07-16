# Sandbox image for cbias_config.py (numpy/pandas/matplotlib/openpyxl only - no bioimage libs).
# Build with: docker build --target cbias-analysis -t cbias-analysis:latest .
FROM python:3.13-slim AS cbias-analysis

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Python packages (must match AVAILABLE_LIBRARIES in cbias_config.py)
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    openpyxl

WORKDIR /work


# Default sandbox image for bioimage_config.py.
# Build with: docker build -t bia-analysis:latest .
FROM python:3.13-slim AS bia-analysis

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Python packages (must match AVAILABLE_LIBRARIES in bioimage_config.py)
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