# Render held-out test view 8 of the web 3DGS model at full size and with every Gaussian scaled down (scaling_modifier),
# using the official CUDA rasterizer. Used only for the article cover.
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.expanduser('~/work/ai4e-train/gaussian-splatting'))
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, GaussianModel
from PIL import Image
sys.argv = [sys.argv[0], '-m', os.path.expanduser('~/work/ai4e-train/runs/3dgs-drone-thr6')]
parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
args = get_combined_args(parser); dataset = model.extract(args); dataset.eval = True
g = GaussianModel(dataset.sh_degree); scene = Scene(dataset, g, load_iteration=30000, shuffle=False)
pipe = pipeline.extract(args); cam = scene.getTestCameras()[8]
for modifier in (1.0, 0.2):
    for bg_value in ((0.055, 0.11, 0.153),):
        with torch.no_grad():
            image = render(cam, g, pipe, torch.tensor(bg_value, device='cuda'), scaling_modifier=modifier)['render'].clamp(0, 1)
        Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(os.path.expanduser(f'~/work/ai4e-train/exports/3dgs-inspect/cover-{modifier}.png'))
        print(cam.image_name, modifier, image.shape)
