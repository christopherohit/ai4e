"""Train a small Mixture-of-Experts classifier on the UCI 8×8 digits to look inside a router.

    .venv/bin/python scripts/llm-bake/moe.py

Model: x (64 pixels) → router (64 → 8 experts) → top-2 experts, each an MLP 64 → 32 → 10, outputs mixed by the
renormalised router weights (Mixtral-style). Trained with Adam for 60 epochs, with and without the Switch-Transformer
load-balancing loss  α · E · Σᵢ fᵢ · Pᵢ  (α = 0.02). A dense MLP with the same *active* parameter count is the baseline.
This is a teaching model, not an LLM: it shows routing, specialisation and load balance on data small enough to see.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
torch.manual_seed(0)
E, K, H = 8, 2, 32


class MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(64, E)
        self.w1 = torch.nn.Parameter(torch.randn(E, 64, H) * 0.12)
        self.b1 = torch.nn.Parameter(torch.zeros(E, H))
        self.w2 = torch.nn.Parameter(torch.randn(E, H, 10) * 0.17)
        self.b2 = torch.nn.Parameter(torch.zeros(E, 10))

    def forward(self, x):
        logits = self.router(x)
        probs = logits.softmax(-1)
        top, idx = probs.topk(K, -1)
        gate = top / top.sum(-1, keepdim=True)
        h = torch.relu(torch.einsum('bi,eih->beh', x, self.w1) + self.b1)
        y = torch.einsum('beh,ehc->bec', h, self.w2) + self.b2  # every expert (fine for a tiny model)
        chosen = y.gather(1, idx.unsqueeze(-1).expand(-1, -1, 10))
        out = (gate.unsqueeze(-1) * chosen).sum(1)
        return out, probs, idx, gate, logits


def train(x, y, xt, yt, alpha, epochs=60):
    torch.manual_seed(1)
    model = MoE()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 64):
            b = perm[i:i + 64]
            out, probs, idx, _, _ = model(x[b])
            loss = torch.nn.functional.cross_entropy(out, y[b])
            f = torch.zeros(E).scatter_add_(0, idx.flatten(), torch.ones(idx.numel())) / idx.numel()
            aux = E * (f * probs.mean(0)).sum()
            (loss + alpha * aux).backward()
            opt.step(); opt.zero_grad()
        with torch.no_grad():
            out, probs, idx, _, _ = model(xt)
            usage = torch.bincount(idx.flatten(), minlength=E).float() / idx.numel()
            history.append({'epoch': epoch + 1, 'accuracy': round(float((out.argmax(1) == yt).float().mean()), 4), 'usage': [round(float(u), 4) for u in usage]})
    return model, history


def dense(x, y, xt, yt, epochs=60):
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(64, 2 * H), torch.nn.ReLU(), torch.nn.Linear(2 * H, 10))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 64):
            b = perm[i:i + 64]
            torch.nn.functional.cross_entropy(model(x[b]), y[b]).backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        return round(float((model(xt).argmax(1) == yt).float().mean()), 4), sum(p.numel() for p in model.parameters())


def main():
    d = load_digits()
    X = d.data / 16.0
    x, xt, y, yt, ix, ixt = train_test_split(X, d.target, np.arange(len(X)), test_size=0.3, random_state=0, stratify=d.target)
    x, xt = torch.tensor(x, dtype=torch.float32), torch.tensor(xt, dtype=torch.float32)
    y, yt = torch.tensor(y), torch.tensor(yt)
    out = {'source': 'Optical Recognition of Handwritten Digits (UCI, Alpaydin & Kaynak 1998), bản 8×8 của scikit-learn: 1 257 ảnh huấn luyện, 540 ảnh kiểm tra.',
           'experts': E, 'topK': K, 'hidden': H}
    for name, alpha in [('balanced', 0.02), ('unbalanced', 0.0)]:
        model, history = train(x, y, xt, yt, alpha)
        with torch.no_grad():
            o, probs, idx, gate, logits = model(xt)
        heat = np.zeros((10, E))
        for c in range(10):
            sel = idx[yt == c].flatten()
            heat[c] = np.bincount(sel.numpy(), minlength=E) / max(1, sel.numel())
        samples = []
        for c in range(10):
            j = int(np.flatnonzero(yt.numpy() == c)[0])
            samples.append({'label': c, 'pixels': (xt[j] * 16).round().int().tolist(), 'routerProbs': [round(float(v), 4) for v in probs[j]],
                            'experts': idx[j].tolist(), 'gates': [round(float(v), 4) for v in gate[j]], 'pred': int(o[j].argmax())})
        expert_params = 64 * H + H + H * 10 + 10
        out[name] = {'alpha': alpha, 'history': history, 'accuracy': history[-1]['accuracy'], 'heat': np.round(heat, 4).tolist(), 'samples': samples,
                     'totalParams': sum(p.numel() for p in model.parameters()), 'activeParams': 64 * E + E + K * expert_params}
        print(name, history[-1])
    acc, params = dense(x, y, xt, yt)
    out['dense'] = {'accuracy': acc, 'params': params}
    print('dense', acc, params, 'moe active', out['balanced']['activeParams'], 'total', out['balanced']['totalParams'])
    path = ROOT / 'tools/kien-thuc/moe-explainer/data/moe.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(path, path.stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
