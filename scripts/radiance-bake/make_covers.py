# Article covers for the four radiance labs, built only from real outputs of the trained models (1600x900 WebP).
# Faces are never shown as texture (user decision 2026-10-09): SyncTalk uses depth + real ray traces,
# InsTaG uses the real Δμ field over the face silhouette. No text: the card title sits under the image.
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

EXPORTS = os.path.expanduser('~/work/ai4e-train/exports')
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/Documents/ai4e/blog/src/assets/covers')
W, H = 1600, 900
INK = (14, 28, 39)

def cover_fit(image, width=W, height=H):
    scale = max(width / image.width, height / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    left = (image.width - width) // 2; top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))

def vignette(image, strength=0.55):
    y, x = np.mgrid[0:H, 0:W]
    distance = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    mask = np.clip(1 - strength * np.clip(distance - 0.35, 0, None), 0, 1)[..., None]
    array = np.asarray(image).astype(np.float32)
    return Image.fromarray((array * mask + np.array(INK) * (1 - mask)).astype(np.uint8))

# 1. NeRF: one orbit frame of the trained nerfacto field; a diagonal band reveals its depth map (same frame).
def diagonal_mask(position=0.5, softness=0.06, slope=0.35):
    y, x = np.mgrid[0:H, 0:W]
    field = x / W + (y / H - 0.5) * slope
    return Image.fromarray((np.clip((field - position) / softness + 0.5, 0, 1) * 255).astype(np.uint8))

def colormap(gray):
    # Perceptual dark-teal → sky → pale ramp for depth, close to the blog palette.
    stops = np.array([[8, 22, 36], [12, 74, 110], [14, 165, 233], [186, 230, 253]], np.float32)
    values = np.asarray(gray).astype(np.float32) / 255 * (len(stops) - 1)
    low = np.floor(values).astype(int).clip(0, len(stops) - 2); t = (values - low)[..., None]
    return Image.fromarray((stops[low] * (1 - t) + stops[low + 1] * t).astype(np.uint8))

def nerf():
    colour = cover_fit(Image.open(f'{EXPORTS}/nerf-drone/frame-04.webp').convert('RGB'))
    depth = cover_fit(Image.open(f'{EXPORTS}/nerf-drone/depth-04.webp').convert('RGB')).convert('L')
    image = Image.composite(colormap(Image.eval(depth, lambda v: 255 - v)), colour, diagonal_mask(0.58))
    return vignette(image, 0.3)

# 2. 3DGS: the trained model at a held-out test view, rendered by the CUDA rasterizer; the left part uses
# scaling_modifier 0.2 so the individual Gaussians show, the right part is the normal render.
def gaussian():
    full = cover_fit(Image.open(f'{EXPORTS}/3dgs-inspect/cover-1.0.png').convert('RGB'))
    shrunk = cover_fit(Image.open(f'{EXPORTS}/3dgs-inspect/cover-0.2.png').convert('RGB'))
    return vignette(Image.composite(full, shrunk, diagonal_mask(0.46)), 0.3)

# 3. SyncTalk: the trained head field seen only through its rays (90×90 grid baked for the cover). Each ray is a
# short horizontal stroke at its pixel; along the stroke, position = depth t (near → far) and brightness = weight
# w = T·α. No colour texture: the face reads as a relief made of real volume-rendering samples.
def synctalk():
    meta = json.load(open(f'{EXPORTS}/synctalk-cover/meta.json')); columns, rows = meta['grid']
    traces = np.fromfile(f'{EXPORTS}/synctalk-cover/traces-0100.f32', np.float32).reshape(columns * rows, -1, 8)
    canvas = Image.new('RGB', (W, H), INK); draw = ImageDraw.Draw(canvas, 'RGBA')
    side = 1180; left = (W - side) // 2 + 20; top = (H - side) // 2 + 130; cell = side / columns
    surface = traces[..., 0][np.arange(len(traces)), traces[..., 4].argmax(1)]
    hit = traces[..., 4].sum(1) > 0.2
    near, far = np.percentile(surface[hit], 2), np.percentile(surface[hit], 98)
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if not hit[index]: continue
            samples = traces[index]; cx = left + (column + 0.5) * cell; cy = top + (row + 0.5) * cell
            closeness = 1 - np.clip((surface[index] - near) / max(1e-6, far - near), 0, 1)
            colour = (int(40 + 175 * closeness), int(150 + 95 * closeness), int(225 + 30 * closeness))
            for sample in samples:
                weight = sample[4]
                if weight < 0.01: continue
                x = cx + ((sample[0] - surface[index]) / max(1e-6, far - near)) * cell * 6
                r = 0.9 + 5.0 * weight
                draw.ellipse([x - r, cy - r, x + r, cy + r], fill=colour + (int(140 + 115 * min(1, weight * 3)),))
    return vignette(canvas, 0.2)

# 4. InsTaG: the real Δμ field of the trained motion network, projected through the test camera.
def instag():
    canvas = Image.new('RGB', (W, H), INK)
    alpha = Image.open(f'{EXPORTS}/instag-excerpt/alpha-face-0100.webp').convert('L')
    side = 1080; silhouette = alpha.resize((side, side), Image.LANCZOS)
    offset = ((W - side) // 2, (H - side) // 2 + 40)
    canvas.paste(Image.new('RGB', (side, side), (22, 40, 58)), offset, silhouette.point(lambda v: int(v * 0.85)))
    draw = ImageDraw.Draw(canvas, 'RGBA'); scale = side / 450; magnify = 4
    for name, colour, step, width in (('face', (253, 210, 90), 3, 2), ('mouth', (251, 113, 133), 1, 3)):
        p = np.fromfile(f'{EXPORTS}/instag-excerpt/proj-{name}-0100.f16', np.float16).astype(np.float32).reshape(-1, 4)
        for x0, y0, x1, y1 in p[::step]:
            a = (offset[0] + x0 * scale, offset[1] + y0 * scale)
            b = (a[0] + (x1 - x0) * scale * magnify, a[1] + (y1 - y0) * scale * magnify)
            draw.line([a, b], fill=colour + (170,), width=width)
            draw.ellipse([a[0] - 1.5, a[1] - 1.5, a[0] + 1.5, a[1] + 1.5], fill=colour + (200,))
    return vignette(canvas, 0.2)

os.makedirs(OUT, exist_ok=True)
for name, build in (('nerf', nerf), ('gaussian-splatting', gaussian), ('synctalk', synctalk), ('instag', instag)):
    image = build(); image.save(f'{OUT}/{name}.webp', quality=86, method=6)
    print(name, image.size, os.path.getsize(f'{OUT}/{name}.webp') // 1024, 'KB')
