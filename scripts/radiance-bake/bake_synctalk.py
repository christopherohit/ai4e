# Bake per-pixel ray traces from the trained SyncTalk model for a subset of test frames.
# Reuses SyncTalk's own dataset + model; samples are taken with the same occupancy grid and marching as inference,
# densities/colours come from the trained tri-plane network with the frame's real audio feature and eye (AU45) value.
import json, os, sys
sys.argv = [sys.argv[0], '../data/synctalk/Obama', '--workspace', '../runs/synctalk-obama', '-O', '--test', '--asr_model', 'ave', '--au45'] + sys.argv[1:]
os.chdir(os.path.expanduser('~/work/ai4e-train/SyncTalk')); sys.path.insert(0, '.')
import numpy as np, torch
import raymarching
exec(open('main.py').read().split("    if opt.test:")[0].replace("if __name__ == '__main__':", "if True:"))
from nerf_triplane.utils import Trainer
trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=torch.nn.L1Loss(reduction='none'), fp16=opt.fp16, metrics=[], use_checkpoint='latest')
loader = NeRFDataset(opt, device=device, type='test').dataloader()
model.aud_features = loader._data.auds; model.eye_areas = loader._data.eye_area
out = os.path.expanduser(os.environ.get('OUT', '~/work/ai4e-train/exports/synctalk-obama')); os.makedirs(out, exist_ok=True)
grid = (18, 18); frames = list(range(int(os.environ.get('START', 0)), min(loader._data.poses.shape[0], int(os.environ.get('END', 10**9))), int(os.environ.get('STEP', 25))))
meta = {'width': opt.W, 'height': opt.H, 'grid': grid, 'frames': [], 'traceFields': ['t', 'sigma', 'alpha', 'T', 'weight', 'r', 'g', 'b'], 'maxSamples': 0}
model.eval()
for frame in (frames if os.environ.get('TRACES', '1') == '1' else []):
    data = loader._data.collate([frame])
    rays_o = data['rays_o'][0]; rays_d = data['rays_d'][0]
    xs = ((np.arange(grid[0]) + 0.5) * opt.W / grid[0]).astype(int); ys = ((np.arange(grid[1]) + 0.5) * opt.H / grid[1]).astype(int)
    pick = torch.tensor([y * opt.W + x for y in ys for x in xs], device=device)
    o = rays_o[pick].contiguous(); d = rays_d[pick].contiguous()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=opt.fp16):
        enc_a = model.encode_audio(data['auds'])
        eye = data['eye']
        nears, fars = raymarching.near_far_from_aabb(o, d, model.aabb_infer, model.min_near)
        traces = []
        for ray in range(o.shape[0]):
            near, far = float(nears[ray]), float(fars[ray])
            t = torch.linspace(near, far, 96, device=device)
            xyz = (o[ray][None] + d[ray][None] * t[:, None])
            dirs = d[ray][None].expand_as(xyz)
            sigma, rgb, _, _, _ = model(xyz, dirs, enc_a, model.individual_codes[0][None] if model.individual_dim > 0 else None, eye)
            sigma = (model.density_scale * sigma).float(); rgb = rgb.float()
            delta = torch.full_like(t, (far - near) / 96)
            alpha = 1 - torch.exp(-sigma * delta)
            T = torch.cumprod(torch.cat([torch.ones(1, device=device), 1 - alpha[:-1]]), 0)
            w = alpha * T
            traces.append(torch.stack([t, sigma, alpha, T, w, rgb[:, 0], rgb[:, 1], rgb[:, 2]], -1).cpu().numpy())
    arr = np.stack(traces).astype(np.float32); arr.tofile(f'{out}/traces-{frame:04d}.f32')
    meta['frames'].append({'frame': frame, 'eye': float(eye.item()) if eye is not None else None, 'samples': arr.shape[1]})
    meta['maxSamples'] = arr.shape[1]
    print('frame', frame, arr.shape, 'sum w mean', float(arr[..., 4].sum(1).mean()), flush=True)
if os.environ.get('TRACES', '1') == '1': json.dump(meta, open(f'{out}/meta.json', 'w'))

# Head-only render (no torso/background composite) and depth for the same baked frames, at full 450x450, from the
# model's own CUDA ray marcher. Also cross-checks the 96-sample uniform traces against the CUDA marcher's colour.
if os.environ.get('RENDER', '1') == '1':
    from PIL import Image
    checks = []
    for frame in frames:
        data = loader._data.collate([frame])
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=opt.fp16):
            black = torch.zeros(1, opt.H * opt.W, 3, device=device)
            result = model.render(data['rays_o'], data['rays_d'], data['auds'], data['bg_coords'], data['poses'], eye=data['eye'], index=data['index'], staged=True, bg_color=black, perturb=False, **{k: v for k, v in vars(opt).items() if k not in ('bg_color',)})
        image = result['image'].reshape(opt.H, opt.W, 3).float().clamp(0, 1).cpu().numpy()
        weights = result['weights_sum'].reshape(opt.H, opt.W).float().clamp(0, 1).cpu().numpy() if 'weights_sum' in result else None
        depth = result['depth'].reshape(opt.H, opt.W).float().cpu().numpy()
        Image.fromarray((image * 255).astype(np.uint8)).save(f'{out}/head-{frame:04d}.webp', quality=90)
        if weights is not None: Image.fromarray((weights * 255).astype(np.uint8)).save(f'{out}/alpha-{frame:04d}.webp', quality=85)
        valid = depth[depth > 0]; near, far = (np.percentile(valid, 2), np.percentile(valid, 98)) if valid.size else (0, 1)
        Image.fromarray((np.clip((depth - near) / max(1e-6, far - near), 0, 1) * 255 * (depth > 0)).astype(np.uint8)).save(f'{out}/depth-{frame:04d}.webp', quality=85)
        traces = np.fromfile(f'{out}/traces-{frame:04d}.f32', np.float32).reshape(grid[0] * grid[1], -1, 8)
        xs = ((np.arange(grid[0]) + 0.5) * opt.W / grid[0]).astype(int); ys = ((np.arange(grid[1]) + 0.5) * opt.H / grid[1]).astype(int)
        cuda = np.array([image[y, x] for y in ys for x in xs]); ours = (traces[..., 4:5] * traces[..., 5:8]).sum(1)
        checks.append(float(np.abs(cuda - ours).mean()))
    meta = json.load(open(f'{out}/meta.json')); meta['headRender'] = {'files': 'head-XXXX.webp, alpha-XXXX.webp, depth-XXXX.webp', 'meanAbsDiffTracesVsCuda': round(float(np.mean(checks)), 4), 'note': 'traces use 96 uniform samples in aabb_infer; CUDA uses occupancy-grid marching'}
    json.dump(meta, open(f'{out}/meta.json', 'w')); print('trace vs cuda mean abs diff', np.mean(checks))
