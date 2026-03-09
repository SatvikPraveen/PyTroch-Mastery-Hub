# PyTorch Mastery Hub - Development Container
FROM python:3.10-slim

# Metadata
LABEL maintainer="PyTorch Mastery Hub Contributors"
LABEL description="Development environment for PyTorch Mastery Hub"
LABEL version="1.0.0"

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

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install the package in editable mode
COPY src/ ./src/
RUN pip install -e ".[dev,notebooks,advanced]"

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
