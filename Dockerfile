# PyTorch Mastery Hub - Development Container
FROM python:3.12-slim

# Metadata
LABEL maintainer="Satvik Praveen <satvikpraveen707@gmail.com>"
LABEL description="Development environment for PyTorch Mastery Hub"
LABEL version="1.1.1"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install dependencies first (cached layer), CPU torch wheels keep the image small
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev,notebooks]"

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/data \
             /workspace/models \
             /workspace/outputs \
             /workspace/logs

# Expose Jupyter port
EXPOSE 8888

# Expose TensorBoard port
EXPOSE 6006

# Set up Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
