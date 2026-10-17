# For the web 3DGS drone model: pick a grid of pixels on a few held-out test views and record, per pixel,
# the depth-sorted Gaussians that contribute, with alpha, transmittance T and weight w = alpha*T,
# computed exactly like the CUDA forward pass (EWA projection + 0.3 low-pass, alpha clamp 0.99, skip alpha<1/255,
# stop when T < 1e-4). We cross-check the composited colour against the CUDA rasterizer output.
import json, math, os, sys, numpy as np, torch
sys.path.insert(0, os.path.expanduser('~/work/ai4e-train/gaussian-splatting'))
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.sh_utils import eval_sh
from PIL import Image
sys.argv = [sys.argv[0], '-m', os.path.expanduser('~/work/ai4e-train/runs/3dgs-drone-thr6')]
parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
args = get_combined_args(parser); dataset = model.extract(args); dataset.eval = True
g = GaussianModel(dataset.sh_degree); scene = Scene(dataset, g, load_iteration=30000, shuffle=False)
pipe = pipeline.extract(args); bg = torch.zeros(3, device='cuda')
out = os.path.expanduser('~/work/ai4e-train/exports/3dgs-inspect'); os.makedirs(out, exist_ok=True)
xyz = g.get_xyz; opacity = g.get_opacity[:, 0]; cov3 = g.get_covariance()  # [N,6] upper triangle
views = scene.getTestCameras(); pick_views = [0, 8, 16, 24]; grid = (16, 10); meta = {'grid': grid, 'views': []}
for vi in pick_views:
    cam = views[vi]; W, H = cam.image_width, cam.image_height
    with torch.no_grad():
        rendered = torch.clamp(render(cam, g, pipe, bg)['render'], 0, 1)
    Image.fromarray((rendered.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).resize((960, int(960 * H / W))).save(f'{out}/view-{vi:02d}.webp', quality=88)
    Image.fromarray((cam.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).resize((960, int(960 * H / W))).save(f'{out}/gt-{vi:02d}.webp', quality=88)
    view = cam.world_view_transform.T  # world->camera (column-major in 3DGS)
    p_cam = (xyz @ view[:3, :3].T) + view[:3, 3]
    fx = W / (2 * math.tan(cam.FoVx / 2)); fy = H / (2 * math.tan(cam.FoVy / 2))
    z = p_cam[:, 2]; front = z > 0.2
    tanx, tany = math.tan(cam.FoVx / 2), math.tan(cam.FoVy / 2)
    tx = (p_cam[:, 0] / z).clamp(-1.3 * tanx, 1.3 * tanx) * z; ty = (p_cam[:, 1] / z).clamp(-1.3 * tany, 1.3 * tany) * z
    J = torch.zeros(len(z), 2, 3, device='cuda')
    J[:, 0, 0] = fx / z; J[:, 0, 2] = -fx * tx / z**2; J[:, 1, 1] = fy / z; J[:, 1, 2] = -fy * ty / z**2
    S = torch.zeros(len(z), 3, 3, device='cuda'); c = cov3
    S[:, 0, 0], S[:, 0, 1], S[:, 0, 2], S[:, 1, 1], S[:, 1, 2], S[:, 2, 2] = c[:, 0], c[:, 1], c[:, 2], c[:, 3], c[:, 4], c[:, 5]
    S[:, 1, 0], S[:, 2, 0], S[:, 2, 1] = c[:, 1], c[:, 2], c[:, 4]
    Wm = view[:3, :3]; T = J @ Wm
    cov2 = T @ S @ T.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3; b = cov2[:, 0, 1]; d = cov2[:, 1, 1] + 0.3
    det = a * d - b * b; conic = torch.stack([d / det, -b / det, a / det], -1)
    mean2 = torch.stack([fx * p_cam[:, 0] / z + W / 2 - 0.5, fy * p_cam[:, 1] / z + H / 2 - 0.5], -1)
    dirs = xyz - cam.camera_center; dirs = dirs / dirs.norm(dim=1, keepdim=True)
    shs = g.get_features.transpose(1, 2)
    colour = torch.clamp_min(eval_sh(g.active_sh_degree, shs, dirs) + 0.5, 0)
    order = torch.argsort(z)
    pixels = []
    for py in ((np.arange(grid[1]) + 0.5) * H / grid[1]).astype(int):
        for px in ((np.arange(grid[0]) + 0.5) * W / grid[0]).astype(int):
            dx = mean2[order, 0] - px; dy = mean2[order, 1] - py
            power = -0.5 * (conic[order, 0] * dx * dx + conic[order, 2] * dy * dy) - conic[order, 1] * dx * dy
            alpha = torch.clamp(opacity[order] * torch.exp(power), max=0.99)
            valid = front[order] & (power <= 0) & (alpha >= 1 / 255) & (det[order] > 0)
            idx = torch.nonzero(valid)[:, 0]
            al = alpha[idx]; Ts = torch.cumprod(torch.cat([torch.ones(1, device='cuda'), 1 - al[:-1]]), 0)[:len(al)]
            keep = Ts >= 1e-4; al, Ts, idx = al[keep], Ts[keep], idx[keep]
            w = al * Ts; col = (w[:, None] * colour[order][idx]).sum(0)
            ref = rendered[:, py, px]
            top = torch.argsort(w, descending=True)[:64]; top = top.sort().values
            pixels.append({'x': int(px), 'y': int(py), 'n': int(len(idx)), 'rgb': [round(float(v), 4) for v in col], 'cudaRgb': [round(float(v), 4) for v in ref],
                           'samples': [[round(float(z[order][idx][k]), 4), round(float(opacity[order][idx][k]), 4), round(float(al[k]), 4), round(float(Ts[k]), 5), round(float(w[k]), 5)] + [round(float(v), 3) for v in colour[order][idx][k]] for k in top.tolist()]})
    err = float(np.mean([np.abs(np.array(p['rgb']) - np.array(p['cudaRgb'])).max() for p in pixels]))
    meta['views'].append({'index': vi, 'name': cam.image_name, 'width': W, 'height': H, 'maxAbsDiffVsCuda': err})
    json.dump(pixels, open(f'{out}/pixels-{vi:02d}.json', 'w'))
    print('view', vi, cam.image_name, 'pixels', len(pixels), 'mean max|rgb-cuda|', round(err, 4), flush=True)
meta['sampleFields'] = ['depth', 'opacity', 'alpha', 'T', 'weight', 'r', 'g', 'b']
json.dump(meta, open(f'{out}/meta.json', 'w'))
