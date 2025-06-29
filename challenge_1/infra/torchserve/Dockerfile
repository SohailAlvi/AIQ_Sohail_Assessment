FROM pytorch/torchserve:latest

# Install required Python dependencies (OpenCV, torchvision, etc.)
RUN pip install --no-cache-dir opencv-python-headless torchvision numpy pillow

# Set working directory (TorchServe expects this location)
WORKDIR /home/model-server/

# Optional: Disable TorchServe auth by default
ENV TS_AUTH_ENABLED=false

# Expose ports for inference and management APIs
EXPOSE 8080 8081

# Start TorchServe with your model and config
CMD ["torchserve", "--start", "--disable-token-auth", "--model-store", "/home/model-server/model-store", "--models", "maskrcnn_v2=maskrcnn_v2.mar", "--ncs"]
