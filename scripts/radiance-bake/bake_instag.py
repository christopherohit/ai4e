# Bake InsTaG (trained on Obama) for the browser, using InsTaG's own fuse checkpoint and renderers:
#  - canonical face + mouth Gaussians (SH1) as PLY -> SPZ later
#  - for sampled test frames: per-Gaussian displacement d_xyz from the trained motion networks (float16),
#    the frame's audio/AU condition, and the fused render + face/mouth alpha (webp) from render_motion*.
import copy, json, os, sys
os.chdir(os.path.expanduser('~/work/ai4e-train/InsTaG')); sys.path.insert(0, '.')
sys.argv = [sys.argv[0], '-s', 'data/Obama', '-m', '../runs/instag-obama', '--eval', '--audio_extractor', 'ave']
import numpy as np, torch
from PIL import Image
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork, MouthMotionNetwork
from gaussian_renderer import render_motion, render_motion_mouth_con
parser = ArgumentParser(); model = ModelParams(parser); pipeline = PipelineParams(parser)
args = get_combined_args(parser); dataset = model.extract(args); pipe = pipeline.extract(args)
out = os.path.expanduser(os.environ.get('OUT', '~/work/ai4e-train/exports/instag-obama')); os.makedirs(out, exist_ok=True)
with torch.no_grad():
    dataset.type = 'face'; face = GaussianModel(copy.deepcopy(dataset))
    dataset.type = 'mouth'; mouth = GaussianModel(copy.deepcopy(dataset))
    scene = Scene(dataset, face, shuffle=False)
    net = MotionNetwork(args=dataset).cuda(); net_mouth = MouthMotionNetwork(args=dataset).cuda()
    model_params, motion_params, mouth_params, motion_mouth_params = torch.load(os.path.join(dataset.model_path, 'chkpnt_fuse_latest.pth'), weights_only=False)
    net.load_state_dict(motion_params, strict=False); face.restore(model_params, None)
    net_mouth.load_state_dict(motion_mouth_params, strict=False); mouth.restore(mouth_params, None)
    face.save_ply(f'{out}/face-canonical.ply'); mouth.save_ply(f'{out}/mouth-canonical.ply')
    background = torch.zeros(3, device='cuda')
    views = scene.getTestCameras(); step = int(os.environ.get('STEP', 25))
    meta = {'faceCount': int(face.get_xyz.shape[0]), 'mouthCount': int(mouth.get_xyz.shape[0]), 'frames': []}
    for index in range(int(os.environ.get('START', 0)), min(len(views), int(os.environ.get('END', 10**9))), step):
        view = views[index]
        pkg = render_motion(view, face, net, pipe, background, frame_idx=0, align=True)
        pkg_mouth = render_motion_mouth_con(view, mouth, net_mouth, face, net, pipe, background, frame_idx=0, align=True, inference=True)
        mouth_image = pkg_mouth['render'] + view.background.cuda() / 255.0 * (1.0 - pkg_mouth['alpha'])
        image = pkg['render'] + mouth_image * (1.0 - pkg['alpha'])
        Image.fromarray((image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f'{out}/frame-{index:04d}.webp', quality=90)
        Image.fromarray((pkg['alpha'][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(f'{out}/alpha-face-{index:04d}.webp', quality=85)
        Image.fromarray((pkg_mouth['alpha'][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(f'{out}/alpha-mouth-{index:04d}.webp', quality=85)
        d_face = pkg['motion']['d_xyz'] if 'motion' in pkg else None
        d_mouth = pkg_mouth['motion']['d_xyz'] if 'motion' in pkg_mouth else None
        for name, value in (('face', d_face), ('mouth', d_mouth)):
            if value is not None: value.half().cpu().numpy().tofile(f'{out}/dxyz-{name}-{index:04d}.f16')
        # Project canonical mu and deformed mu + d_mu through the real test camera (InsTaG full_proj_transform),
        # so the browser can draw displacement vectors exactly over the rendered face. Pixel units, float16.
        for name, model, package in (('face', face, pkg), ('mouth', mouth, pkg_mouth)):
            if 'motion' not in package: continue
            base = model.get_xyz; moved = base + package['motion']['d_xyz']
            def to_pixels(points):
                homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], 1) @ view.full_proj_transform
                ndc = homogeneous[:, :2] / homogeneous[:, 3:4]
                return torch.stack([(ndc[:, 0] + 1) * view.image_width / 2 - 0.5, (ndc[:, 1] + 1) * view.image_height / 2 - 0.5], 1)
            torch.cat([to_pixels(base), to_pixels(moved)], 1).half().cpu().numpy().tofile(f'{out}/proj-{name}-{index:04d}.f16')
        talking = view.talking_dict
        meta['frames'].append({'index': index, 'imageName': view.image_name, 'imgId': int(talking['img_id']),
                               'faceMotion': d_face is not None, 'mouthMotion': d_mouth is not None,
                               'meanDisplacement': {'face': float(d_face.norm(dim=-1).mean()) if d_face is not None else None, 'mouth': float(d_mouth.norm(dim=-1).mean()) if d_mouth is not None else None}})
        print('frame', index, view.image_name, meta['frames'][-1]['meanDisplacement'], flush=True)
    meta['width'] = int(views[0].image_width); meta['height'] = int(views[0].image_height)
    json.dump(meta, open(f'{out}/meta.json', 'w'))
