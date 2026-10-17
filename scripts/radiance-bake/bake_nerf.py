# Bake the trained nerfacto drone model for the browser:
#  - an orbit of RGB + depth frames (WebP) around the scene, camera poses in the dataparser frame
#  - for a grid of inspectable pixels per frame, the real per-sample trace (t, sigma, alpha, T, weight, rgb)
# All values come from the trained field; nothing is approximated in the browser.
import functools, json, math, os, sys
import numpy as np, torch
torch.load = functools.partial(torch.load, weights_only=False)  # our own checkpoint
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.field_components.field_heads import FieldHeadNames
from PIL import Image

def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    frames_count = int(os.environ.get('FRAMES', 36)); width, height = 960, 600; grid = (24, 15)
    _, pipeline, _, _ = eval_setup(Path(os.path.expanduser('~/work/ai4e-train/runs/nerf-drone/nerfacto/main/config.yml')), test_mode='inference')
    model = pipeline.model.eval()
    train = pipeline.datamanager.train_dataset.cameras
    poses = train.camera_to_worlds
    positions = poses[:, :3, 3]; forwards = -poses[:, :3, 2]
    # Look-at target: least-squares point closest to all training view rays (what the drone was filming).
    eye_matrix = torch.eye(3)[None] - forwards[:, :, None] * forwards[:, None, :]
    target = torch.linalg.solve(eye_matrix.sum(0), (eye_matrix @ positions[:, :, None]).sum(0))[:, 0]
    offsets = positions - target
    radius = float(offsets[:, :2].norm(dim=1).median()); lift = float(offsets[:, 2].median())
    focal = float(train.fx[0]) * width / float(train.width[0])
    meta = {'width': width, 'height': height, 'grid': grid, 'focal': focal, 'target': target.tolist(), 'radius': radius, 'lift': lift, 'frames': []}
    for index in range(frames_count):
        angle = 2 * math.pi * index / frames_count
        eye = target + torch.tensor([radius * math.cos(angle), radius * math.sin(angle), lift])
        forward = (target - eye); forward = forward / forward.norm()
        right = torch.cross(forward, torch.tensor([0.0, 0.0, 1.0]), dim=0); right = right / right.norm()
        up = torch.cross(right, forward, dim=0)
        c2w = torch.stack([right, up, -forward, eye], dim=1)[None]
        camera = Cameras(camera_to_worlds=c2w, fx=focal, fy=focal, cx=width / 2, cy=height / 2, width=width, height=height, camera_type=CameraType.PERSPECTIVE).to(model.device)
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera)
        rgb = (outputs['rgb'].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        depth = outputs['depth'][..., 0].cpu().numpy()
        Image.fromarray(rgb).save(out / f'frame-{index:02d}.webp', quality=88)
        near, far = np.percentile(depth, 2), np.percentile(depth, 98)
        Image.fromarray((np.clip((depth - near) / (far - near), 0, 1) * 255).astype(np.uint8)).save(out / f'depth-{index:02d}.webp', quality=85)
        # per-pixel traces on a grid
        xs = (np.arange(grid[0]) + 0.5) * width / grid[0]; ys = (np.arange(grid[1]) + 0.5) * height / grid[1]
        coords = torch.tensor([[y, x] for y in ys for x in xs], dtype=torch.float32, device=model.device)
        bundle = camera.generate_rays(camera_indices=0, coords=coords)
        bundle = model.collider(bundle)  # same near/far as get_outputs_for_camera
        with torch.no_grad():
            samples, _, _ = model.proposal_sampler(bundle, density_fns=model.density_fns)
            field = model.field.forward(samples)
            sigma = field[FieldHeadNames.DENSITY][..., 0]
            deltas = samples.deltas[..., 0]
            alpha = 1 - torch.exp(-sigma * deltas)
            transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=1), dim=1)
            weight = alpha * transmittance
            colour = field[FieldHeadNames.RGB]
            t = ((samples.frustums.starts + samples.frustums.ends) / 2)[..., 0]
        traces = np.stack([t.cpu().numpy(), sigma.cpu().numpy(), alpha.cpu().numpy(), transmittance.cpu().numpy(), weight.cpu().numpy(), *colour.permute(2, 0, 1).cpu().numpy()], axis=-1).astype(np.float32)
        traces.tofile(out / f'traces-{index:02d}.f32')
        meta['frames'].append({'index': index, 'azimuth': round(math.degrees(angle), 2), 'depthRange': [float(near), float(far)], 'samples': int(traces.shape[1])})
        print('frame', index, rgb.shape, traces.shape, flush=True)
    meta['traceFields'] = ['t', 'sigma', 'alpha', 'T', 'weight', 'r', 'g', 'b']
    (out / 'meta.json').write_text(json.dumps(meta))

if __name__ == '__main__':
    main()
