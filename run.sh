export LOCAL_RANK=0
python -m paddle.distributed.launch --gpus "0,1" train.py