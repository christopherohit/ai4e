# Recreate SyncTalk's <id>_face.png from the existing face-parsing colour maps (data_utils/face_parsing/test.py logic).
# Existing parsing PNGs (RGB): blue (0,0,255)=face skin incl. ears, grey (100,100,100)=mouth, black=hair,
# green=neck, red=torso, white=background. SyncTalk's face mask is face skin + mouth without hair, extended
# slightly below the jaw; the provider reads only the green channel, so the output colour is red-on-white.
import os, sys, cv2, numpy as np
from concurrent.futures import ProcessPoolExecutor
root = sys.argv[1]; out = sys.argv[2]; os.makedirs(out, exist_ok=True)
def one(name):
    target = os.path.join(out, name.replace('.png', '_face.png'))
    if os.path.exists(target): return
    bgr = cv2.imread(os.path.join(root, name))
    face = ((bgr[..., 0] == 255) & (bgr[..., 1] == 0) & (bgr[..., 2] == 0)) | ((bgr[..., 0] == 100) & (bgr[..., 1] == 100) & (bgr[..., 2] == 100))
    vis = np.full(bgr.shape, 255, np.uint8); vis[face] = (0, 0, 255)
    coords = np.stack(np.nonzero(face), -1)
    if len(coords):
        pad = 5; order = np.lexsort((-coords[:, 0], coords[:, 1])); coords = coords[order]
        _, first = np.unique(coords[:, 1], return_index=True); bottom = coords[first] + np.array([pad, 0])
        bottom[:, 0] = np.clip(bottom[:, 0], 0, vis.shape[0] - 1)
        low, high = bottom[:, 1].min(), bottom[:, 1].max(); step = (high - low) // 4
        for x, y in bottom:
            if low + step <= y <= low + 3 * step: vis[max(x - pad, 0):min(x + pad, vis.shape[0]), y] = (0, 0, 255)
    cv2.imwrite(target, cv2.GaussianBlur(vis, (9, 9), cv2.BORDER_DEFAULT))
names = [n for n in os.listdir(root) if n.endswith('.png') and '_' not in n]
with ProcessPoolExecutor(16) as pool: list(pool.map(one, names, chunksize=64))
print(len(names), 'masks')
