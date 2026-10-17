#!/bin/bash
# InsTaG scripts/pretrain_con_ave.sh, on GPU 0, output to ../runs/instag-pretrain-ave
set -e
cd ~/work/ai4e-train/InsTaG
export CUDA_VISIBLE_DEVICES=0
PY=../.venv-tag/bin/python
$PY pretrain_face.py -s data/pretrain -m ../runs/instag-pretrain-ave --type face --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor ave --iterations 30000
$PY pretrain_mouth.py -s data/pretrain -m ../runs/instag-pretrain-ave --type mouth --init_num 5000 --audio_extractor ave --iterations 30000
echo PRETRAIN_DONE
