#!/bin/bash

ROOT_DIR=$(dirname $(dirname $(realpath $0)))
DEVICE=$1

pip install -r "$ROOT_DIR/requirements.txt"

if [[ $DEVICE == "gpu" ]]; then
  pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
else
  pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi
