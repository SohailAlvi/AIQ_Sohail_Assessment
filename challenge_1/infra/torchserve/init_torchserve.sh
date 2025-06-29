docker run -d --name torchserve \
  -p 8080:8080 \
  -p 8081:8081 \
  -v "model-store":"/home/model-server/model-store" \
  torchserve-with-opencv
