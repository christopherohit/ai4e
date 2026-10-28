"""How much of the real attention mass do sparse patterns keep? Measured on Qwen2.5-0.5B-Instruct, CPU.

    .venv/bin/python scripts/llm-bake/sparse_attention.py

One forward pass over 1 024 tokens of WikiText-2 (test split) with eager attention. For every layer and query head, and
for the queries in the second half of the sequence (positions 512–1023), the softmax attention row is compared with
the token sets each pattern would read:
  local W      the last W tokens
  sink + local the first 4 tokens plus the last W − 4
  block top-k  QSA-style: keys mean-pooled into blocks of 4, the query scores every complete block (q · mean k),
               keeps the best k blocks (budget B = 4k tokens) plus the current partial block
  oracle top-k the k blocks with the largest true attention mass (an upper bound for any block selector)
Coverage = attention mass (from the dense softmax) that falls inside the kept set. Also exported: one attention map
(64 × 64 average-pooled) and one query row per selected head, for the lab's pictures.
"""
import json
from pathlib import Path

import numpy as np
import torch
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/llm-bake'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
SEQ, BLOCK, SINK = 1024, 4, 4
WINDOWS = [64, 128, 256]
BUDGETS = [64, 128, 256]
QUERIES = torch.arange(SEQ // 2, SEQ)
torch.set_num_threads(4)

stats = []    # per layer: per head dict of coverages
pictures = {}  # (layer, head) -> map + row
SHOW = [(0, 0), (6, 3), (13, 5), (23, 10)]


def coverage(A, q, k):
    """A: [heads, SEQ, SEQ] softmax rows; q: [heads, SEQ, d]; k: [heads, SEQ, d] (already repeated for GQA)."""
    H = A.shape[0]
    rows = A[:, QUERIES]                                  # [H, Q, SEQ]
    pos = torch.arange(SEQ)
    out = {}
    for W in WINDOWS:
        local = (pos[None, :] > QUERIES[:, None] - W) & (pos[None, :] <= QUERIES[:, None])
        sink_local = ((pos[None, :] > QUERIES[:, None] - (W - SINK)) & (pos[None, :] <= QUERIES[:, None])) | (pos[None, :] < SINK)
        out[f'local{W}'] = (rows * local).sum(-1).mean(-1)
        out[f'sink{W}'] = (rows * sink_local).sum(-1).mean(-1)
    nb = SEQ // BLOCK
    block_keys = k.reshape(H, nb, BLOCK, -1).mean(2)        # [H, nb, d]
    scores = torch.einsum('hqd,hbd->hqb', q[:, QUERIES], block_keys)
    mass = rows.reshape(H, len(QUERIES), nb, BLOCK).sum(-1)  # true mass per block
    complete = (QUERIES + 1) // BLOCK                        # blocks fully before the query's own block
    valid = torch.arange(nb)[None, :] < complete[:, None]    # [Q, nb]
    own = torch.arange(nb)[None, :] == (QUERIES // BLOCK)[:, None]
    for B in BUDGETS:
        kb = B // BLOCK
        for name, s in (('index', scores), ('oracle', mass)):
            s = s.masked_fill(~valid[None], float('-inf'))
            top = s.topk(kb, dim=-1).indices
            keep = torch.zeros_like(mass, dtype=torch.bool).scatter_(-1, top, True) & valid[None]
            keep |= own[None]
            out[f'{name}{B}'] = (mass * keep).sum(-1).mean(-1)
    return {key: [round(float(v), 4) for v in val] for key, val in out.items()}


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    k = qwen2.repeat_kv(key, module.num_key_value_groups)
    v = qwen2.repeat_kv(value, module.num_key_value_groups)
    w = torch.matmul(query, k.transpose(2, 3)) * scaling
    causal = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), 1)
    w = w.masked_fill(causal, float('-inf'))
    A = torch.softmax(w.float(), dim=-1)
    layer = module.layer_idx
    stats.append(coverage(A[0], query[0] * scaling, k[0]))
    for (l, h) in SHOW:
        if l == layer:
            pooled = A[0, h].reshape(64, SEQ // 64, 64, SEQ // 64).mean((1, 3))
            pictures[f'{l}-{h}'] = {'layer': l, 'head': h, 'map': pooled.numpy().round(5).tolist(), 'row': A[0, h, SEQ - 1].numpy().round(5).tolist()}
    out = torch.matmul(A.to(v.dtype), v).transpose(1, 2).contiguous()
    return out, None


def main():
    import pyarrow.parquet as pq
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = '\n'.join(pq.read_table(CACHE / 'wikitext2-test.parquet').column('text').to_pylist())
    ids = tok(text, return_tensors='pt').input_ids[:, 5000:5000 + SEQ]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation='eager')
    qwen2.eager_attention_forward = patched
    with torch.no_grad():
        model(ids)
    cfg = model.config
    keys = list(stats[0].keys())
    per_layer = [{k: round(float(np.mean(s[k])), 4) for k in keys} for s in stats]
    overall = {k: round(float(np.mean([v for s in stats for v in s[k]])), 4) for k in keys}
    data = {'model': MODEL, 'device': 'CPU, float32, eager attention', 'source': 'WikiText-2 raw, test split (Merity et al. 2016, CC BY-SA 3.0)',
            'seq': SEQ, 'block': BLOCK, 'sink': SINK, 'windows': WINDOWS, 'budgets': BUDGETS, 'queries': [int(QUERIES[0]), int(QUERIES[-1])],
            'layers': cfg.num_hidden_layers, 'heads': cfg.num_attention_heads, 'kvHeads': cfg.num_key_value_heads,
            'tokens': tok.convert_ids_to_tokens(ids[0].tolist()), 'overall': overall, 'perLayer': per_layer,
            'perHead': [{k: s[k] for k in ('local64', 'sink64', 'index64', 'oracle64')} for s in stats], 'pictures': list(pictures.values())}
    path = ROOT / 'tools/kien-thuc/sparse-attention-explainer/data/attention.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(path, path.stat().st_size // 1024, 'KB')
    print(json.dumps(overall, indent=1))


if __name__ == '__main__':
    main()
