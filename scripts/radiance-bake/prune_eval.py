import os
# Measure 3DGS test PSNR when keeping only the top-K Gaussians (score = opacity * largest axis^2, a screen-coverage proxy)
# and optionally truncating SH to degree 1 (what the browser renderer evaluates).
import sys, torch
sys.path.insert(0, 'gaussian-splatting')
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.image_utils import psnr
parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
args = get_combined_args(parser)
dataset = model.extract(args); dataset.eval = True
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
pipe = pipeline.extract(args); bg = torch.zeros(3, device='cuda')
full = {name: getattr(gaussians, name).detach().clone() for name in ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']}
score = (torch.sigmoid(full['_opacity']).squeeze() * torch.exp(full['_scaling']).max(dim=1).values ** 2)
order = torch.argsort(score, descending=True)
def evaluate(keep, sh1):
    index = order[:keep]
    for name, value in full.items(): setattr(gaussians, name, value[index].clone())
    if sh1: gaussians._features_rest = gaussians._features_rest.clone(); gaussians._features_rest[:, 3:] = 0
    with torch.no_grad():
        values = [psnr(torch.clamp(render(cam, gaussians, pipe, bg)['render'], 0, 1), cam.original_image.cuda()).mean().item() for cam in scene.getTestCameras()]
    return sum(values) / len(values)
total = full['_xyz'].shape[0]
for keep in [int(k) for k in os.environ.get('KEEPS', str(total)).split(',')] or [total]:
    print(f'keep={keep:>9} sh3={evaluate(keep, False):.2f} sh1={evaluate(keep, True):.2f}', flush=True)
