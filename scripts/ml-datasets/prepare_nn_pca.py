"""Data for the activation-function and PCA labs (the two GlassBox AI video topics).

    pip install numpy scikit-learn
    python scripts/ml-datasets/prepare_nn_pca.py

activation-explainer/data/activation.json
  Two-moons points (scikit-learn make_moons, fixed seed) and small networks trained here with full-batch Adam:
  a linear model (logistic regression), 2-3-1 tanh (its 3-D hidden space is drawn in the lab, with snapshots during
  training), 2-8-1 ReLU and a 2-8-1 network with no activation (it collapses to a linear model). Accuracies are on
  a held-out set drawn from the same generator. The XOR network is the hand-written two-ReLU solution.
pca-explainer/data/pca.json
  PCA of Auto MPG weight and horsepower (standardised), and of the scikit-learn 8×8 digits (1 797 images of
  handwritten digits from UCI): mean image, all 64 components, explained variance, 2-D projection of every image and
  64 coefficients for 12 sample images (for reconstruction). Reference values come from sklearn.decomposition.PCA.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, make_moons
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.1f} KB')


def r(a, digits=4):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


ACT = {
    'tanh': (np.tanh, lambda h: 1 - h ** 2),
    'relu': (lambda z: np.maximum(z, 0), lambda h: (h > 0).astype(float)),
    'none': (lambda z: z, lambda h: np.ones_like(h)),
}


def train(x, y, hidden, act, epochs=3000, lr=0.03, seed=0, snapshots=(), decay=0.0):
    """Full-batch Adam on binary cross-entropy (+ optional L2 on weights) for a 2-hidden-1 network."""
    rng = np.random.default_rng(seed)
    params = {'W1': rng.normal(0, 1, (hidden, 2)), 'b1': np.zeros(hidden), 'W2': rng.normal(0, 1 / np.sqrt(hidden), (1, hidden)), 'b2': np.zeros(1)}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    f, df = ACT[act]
    snaps, losses = [], []
    for step in range(1, epochs + 1):
        z1 = x @ params['W1'].T + params['b1']
        h = f(z1)
        p = sigmoid(h @ params['W2'].T + params['b2'])[:, 0]
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        losses.append(loss)
        d2 = (p - y)[:, None] / len(y)
        grads = {'W2': d2.T @ h, 'b2': d2.sum(0)}
        dh = d2 @ params['W2'] * df(h)
        grads['W1'] = dh.T @ x
        grads['b1'] = dh.sum(0)
        grads['W1'] += decay * params['W1']
        grads['W2'] += decay * params['W2']
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            params[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        if step in snapshots:
            snaps.append({'epoch': step, 'loss': round(float(loss), 4), **{k: r(val) for k, val in params.items()}})
    return params, losses, snaps


def predict(params, x, act):
    h = ACT[act][0](x @ params['W1'].T + params['b1'])
    return sigmoid(h @ params['W2'].T + params['b2'])[:, 0]


def activation():
    x, y = make_moons(400, noise=0.12, random_state=7)
    xt, yt = make_moons(2000, noise=0.12, random_state=8)
    center = np.array([0.5, 0.25])
    x, xt = (x - center) / 1.1, (xt - center) / 1.1
    linear = LogisticRegression().fit(x, y)
    snaps = sorted({int(round(v)) for v in np.geomspace(1, 3000, 40)})
    models = {}
    # The tanh network keeps a small weight decay so its hidden units stay out of saturation and the 3-D view is readable.
    for name, hidden, act, decay in [('tanh3', 3, 'tanh', 3e-3), ('relu8', 8, 'relu', 0), ('none8', 8, 'none', 0), ('relu2', 2, 'relu', 0)]:
        best = None
        for seed in range(6):
            params, losses, s = train(x, y, hidden, act, seed=seed, snapshots=snaps if name == 'tanh3' else (), decay=decay)
            acc = float(np.mean((predict(params, xt, act) > 0.5) == yt))
            if best is None or losses[-1] < best[1][-1]:
                best = (params, losses, s, acc, seed)
        params, losses, s, acc, seed = best
        models[name] = {
            'act': act, 'hidden': hidden, 'seed': seed, 'decay': decay, 'testAccuracy': round(acc, 4),
            'trainAccuracy': round(float(np.mean((predict(params, x, act) > 0.5) == y)), 4),
            'loss': [round(float(losses[i]), 4) for i in [int(round(v)) - 1 for v in np.geomspace(1, len(losses), 60)]],
            **{k: r(v) for k, v in params.items()},
        }
        if s:
            models[name]['snapshots'] = s
    write(ROOT / 'tools/kien-thuc/activation-explainer/data/activation.json', {
        'source': 'Two moons sinh bằng scikit-learn make_moons (400 điểm huấn luyện, 2 000 điểm kiểm tra, nhiễu 0,12, seed 7/8).',
        'x': r(x, 3), 'y': y.astype(int).tolist(),
        'linear': {'w': r(linear.coef_[0]), 'b': round(float(linear.intercept_[0]), 4), 'testAccuracy': round(float(linear.score(xt, yt)), 4), 'trainAccuracy': round(float(linear.score(x, y)), 4)},
        'models': models,
    })


def pca():
    mpg = json.loads((ROOT / 'tools/_shared/glass-lab/data/auto-mpg.json').read_text())
    raw = np.column_stack([np.array(mpg['weight']) / 1000, mpg['horsepower']]).astype(float)
    mean, std = raw.mean(0), raw.std(0)
    z = (raw - mean) / std
    cov = np.cov(z, rowvar=False, bias=True)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vecs *= np.sign(vecs[0])
    sk = PCA(2).fit(z)
    assert np.allclose(sk.explained_variance_ * (len(z) - 1) / len(z), vals, atol=1e-9)

    digits = load_digits()
    X, labels = digits.data, digits.target
    p = PCA().fit(X)
    coords = p.transform(X)
    samples = [int(np.flatnonzero(labels == d)[0]) for d in range(10)] + [int(np.flatnonzero(labels == 3)[5]), int(np.flatnonzero(labels == 8)[7])]
    k95 = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.95) + 1)
    write(ROOT / 'tools/kien-thuc/pca-explainer/data/pca.json', {
        'mpg': {
            'source': 'Auto MPG (Quinlan 1993): khối lượng (nghìn lb) và mã lực của 392 xe, chuẩn hóa về trung bình 0, độ lệch chuẩn 1.',
            'mean': r(mean), 'std': r(std), 'cov': r(cov), 'eigenvalues': r(vals), 'eigenvectors': r(vecs.T), 'explained': r(vals / vals.sum()),
        },
        'digits': {
            'source': 'Optical Recognition of Handwritten Digits (UCI, Alpaydin & Kaynak 1998), bản 8×8 đi kèm scikit-learn: 1 797 ảnh, mỗi ảnh 64 điểm ảnh, độ xám 0–16.',
            'mean': r(p.mean_, 3), 'components': r(p.components_, 4), 'explained': r(p.explained_variance_ratio_, 5),
            'coords': r(coords[:, :2], 2), 'labels': labels.tolist(), 'k95': k95,
            'samples': [{'index': i, 'label': int(labels[i]), 'pixels': X[i].astype(int).tolist(), 'coef': r(coords[i], 3)} for i in samples],
        },
    })


if __name__ == '__main__':
    activation()
    pca()
