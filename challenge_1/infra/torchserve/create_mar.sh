# Run from infra/torchserve
mkdir -p model-store

torch-model-archiver \
  --model-name maskrcnn_v2 \
  --version 1.0 \
  --serialized-file ../../app/model/maskrcnn_finetuned_v2.pth \
  --handler ./maskrcnn_handler.py \
  --export-path model-store \
  --extra-files maskrcnn_handler.py

sleep 10
#torchserve --start --model-store model-store --models maskrcnn_v2=maskrcnn_v2.mar --ncs
