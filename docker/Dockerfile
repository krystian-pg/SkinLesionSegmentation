# Use the PyTorch base image
FROM pytorch/pytorch:latest

# Update and install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Creating main directory
WORKDIR /app

# Copying packages to install 
COPY requirements.txt ./

# Installing all necessary Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copying current directory to the main container directory
COPY . .
