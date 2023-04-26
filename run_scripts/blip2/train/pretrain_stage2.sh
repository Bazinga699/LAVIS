#!/bin/sh
date
tar -xf /home/bingxing2/home/scx6150/datasets/coco.tar -C /dev/shm
tar -xf /home/bingxing2/home/scx6150/datasets/VG.tar -C /dev/shm
date
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path /home/bingxing2/home/scx6150/code/LAVIS/lavis/projects/blip2/train/pretrain_stage2.yaml