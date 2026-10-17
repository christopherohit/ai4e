# Pack float32 traces [rays, samples, 8] -> float16, keeping per ray the contiguous sample window that carries
# 99.9% of the ray's weight (plus 2 samples margin). Empty rays keep 0 samples. Output: header + offsets + data.
#   int32 rays, int32 fields, int32 offsets[rays+1] (in samples), float16 samples[total, fields]
import json, sys, numpy as np, pathlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
meta = json.loads((src / 'meta.json').read_text()); fields = len(meta['traceFields'])
grid = meta['grid']; rays = grid[0] * grid[1]
before = after = 0
for path in sorted(src.glob('traces-*.f32')):
    data = np.fromfile(path, np.float32).reshape(rays, -1, fields)
    pieces, offsets = [], [0]
    for ray in data:
        weight = ray[:, 4]; total = weight.sum()
        if total < 1e-4: offsets.append(offsets[-1]); continue
        cumulative = np.cumsum(weight) / total
        start = max(0, int(np.searchsorted(cumulative, 0.0005)) - 2); end = min(len(ray), int(np.searchsorted(cumulative, 0.9995)) + 3)
        pieces.append(ray[start:end]); offsets.append(offsets[-1] + end - start)
    body = np.concatenate(pieces).astype(np.float16) if pieces else np.zeros((0, fields), np.float16)
    out = dst / path.name.replace('.f32', '.bin')
    with open(out, 'wb') as handle:
        np.array([rays, fields], np.int32).tofile(handle); np.array(offsets, np.int32).tofile(handle); body.tofile(handle)
    before += path.stat().st_size; after += out.stat().st_size
meta['traceFormat'] = 'int32 rays, int32 fields, int32 offsets[rays+1], float16 samples[offsets[-1]][fields]; window = 99.9% of ray weight'
(dst / 'meta.json').write_text(json.dumps(meta))
print(f'{before/1e6:.1f} MB -> {after/1e6:.1f} MB')
