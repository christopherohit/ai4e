# ns_data (nerfstudio transforms.json, OpenGL c2w) -> 3DGS Blender layout. Every 8th frame held out for test.
import json, math, os, struct
src = '/shared/datasets/nhanht41/ojt-data/dataset/ns_data'
out = os.path.expanduser('~/work/ai4e-train/data/drone')
data = json.load(open(f'{src}/transforms.json'))
frames = sorted(data['frames'], key=lambda f: f['file_path'])
split = {'train': [], 'test': []}
for index, frame in enumerate(frames):
    stem = os.path.splitext(os.path.basename(frame['file_path']))[0]
    link = f'{out}/images/{stem}.png'
    if not os.path.exists(link): os.symlink(f"{src}/{frame['file_path']}", link)
    split['test' if index % 8 == 0 else 'train'].append({'file_path': f'images/{stem}', 'transform_matrix': frame['transform_matrix']})
angle = 2 * math.atan(frames[0]['w'] / (2 * frames[0]['fl_x']))
for name, items in split.items():
    json.dump({'camera_angle_x': angle, 'frames': items}, open(f'{out}/transforms_{name}.json', 'w'))
# sparse_pc.ply is already in the transforms frame (nerfstudio applies applied_transform); add zero normals for fetchPly.
lines = open(f'{src}/sparse_pc.ply', 'rb').read().split(b'end_header\n', 1)
count = int([l for l in lines[0].split(b'\n') if l.startswith(b'element vertex')][0].split()[2])
rows = lines[1].decode().strip().split('\n')
with open(f'{out}/points3d.ply', 'w') as f:
    f.write(f'ply\nformat ascii 1.0\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    for row in rows:
        x, y, z, r, g, b = row.split()
        f.write(f'{x} {y} {z} 0 0 0 {r} {g} {b}\n')
print(len(split['train']), len(split['test']), count, math.degrees(angle))
