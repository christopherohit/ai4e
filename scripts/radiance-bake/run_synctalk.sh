#!/bin/bash
# SyncTalk README training recipe (ave + OpenFace AU45), Obama, GPU 1.
set -e
cd ~/work/ai4e-train/SyncTalk
export CUDA_VISIBLE_DEVICES=1
PY=../.venv-st/bin/python
$PY main.py ../data/synctalk/Obama --workspace ../runs/synctalk-obama -O --iters 60000 --asr_model ave --au45
$PY main.py ../data/synctalk/Obama --workspace ../runs/synctalk-obama -O --iters 100000 --finetune_lips --patch_size 64 --asr_model ave --au45
$PY main.py ../data/synctalk/Obama --workspace ../runs/synctalk-obama -O --test --asr_model ave --au45
echo SYNCTALK_DONE
