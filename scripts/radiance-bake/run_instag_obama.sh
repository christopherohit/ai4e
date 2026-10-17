#!/bin/bash
# InsTaG adaptation to Obama: scripts/train_xx_few.sh recipe (ave, n_views=250 = 10 s), then render test set + metrics.
set -e
cd ~/work/ai4e-train/InsTaG
export CUDA_VISIBLE_DEVICES=0
PY=../.venv-tag/bin/python
P=../runs/instag-pretrain-ave
W=../runs/instag-obama
$PY train_face.py --type face -s data/Obama -m $W --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor ave --pretrain_path $P/chkpnt_ema_face_latest.pth --iterations 10000 --sh_degree 1 --N_views 250
$PY train_mouth.py --type mouth -s data/Obama -m $W --audio_extractor ave --pretrain_path $P/chkpnt_ema_mouth_latest.pth --init_num 5000 --iterations 10000 --sh_degree 1 --N_views 250
$PY train_fuse_con.py -s data/Obama -m $W --opacity_lr 0.001 --audio_extractor ave --iterations 2000 --sh_degree 1 --N_views 250
$PY synthesize_fuse.py -s data/Obama -m $W --eval --audio_extractor ave --dilate
$PY metrics.py $W/test/ours_None/renders/out.mp4 $W/test/ours_None/gt/out.mp4
echo INSTAG_DONE
