"""Fine-tune Qwen2.5-0.5B-Instruct on SST-2 sentiment with LoRA and its relatives (Hugging Face peft), on CPU.

    .venv/bin/pip install peft pyarrow
    .venv/bin/python scripts/llm-bake/lora.py

Task: "Review: <sentence>\\nSentiment:" → " positive" / " negative" (GLUE SST-2, Socher et al. 2013). Every variant
trains the same 160 steps (batch 8, same examples, same order) on the attention projections q, k, v, o of all 24
layers, and is scored at steps 0, 40, 80 and 160 on 400 validation sentences by comparing the logits of " positive" and " negative".
Variants: LoRA (r = 1, 8, 64), rsLoRA (r = 64), DoRA, PiSSA initialisation, VeRA, LoHa, and a QLoRA-style run whose
frozen base weights are first rounded to 4-bit (RTN, group 64, simulated on CPU; real QLoRA uses NF4 + bitsandbytes).
"""
import json, math, time, urllib.request
from pathlib import Path

import numpy as np
import torch
from peft import LoHaConfig, LoraConfig, VeraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/llm-bake'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
GLUE = 'https://huggingface.co/datasets/nyu-mll/glue/resolve/main/sst2/{split}-00000-of-00001.parquet'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
STEPS, BATCH, EVAL_N, EVAL_AT = 160, 8, 400, [0, 40, 80, 160]
torch.set_num_threads(4)


def sst2(split):
    path = CACHE / f'sst2-{split}.parquet'
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GLUE.format(split=split), path)
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    return list(zip(t.column('sentence').to_pylist(), t.column('label').to_pylist()))


def prompt(s):
    return f'Review: {s.strip()}\nSentiment:'


VARIANTS = [
    ('LoRA r=1', lambda: LoraConfig(r=1, lora_alpha=2, target_modules=TARGETS, lora_dropout=0.0), 2e-4, False),
    ('LoRA r=8', lambda: LoraConfig(r=8, lora_alpha=16, target_modules=TARGETS, lora_dropout=0.0), 2e-4, False),
    ('LoRA r=64', lambda: LoraConfig(r=64, lora_alpha=128, target_modules=TARGETS, lora_dropout=0.0), 2e-4, False),
    ('rsLoRA r=64', lambda: LoraConfig(r=64, lora_alpha=16, use_rslora=True, target_modules=TARGETS, lora_dropout=0.0), 2e-4, False),
    ('DoRA r=8', lambda: LoraConfig(r=8, lora_alpha=16, use_dora=True, target_modules=TARGETS, lora_dropout=0.0), 2e-4, False),
    ('PiSSA r=8', lambda: LoraConfig(r=8, lora_alpha=8, init_lora_weights='pissa', target_modules=TARGETS, lora_dropout=0.0), 2e-4, False),
    ('VeRA r=256', lambda: VeraConfig(r=256, target_modules=TARGETS, vera_dropout=0.0), 4e-3, False),
    ('LoHa r=8', lambda: LoHaConfig(r=8, alpha=16, target_modules=TARGETS), 2e-4, False),
    ('QLoRA-style r=8', lambda: LoraConfig(r=8, lora_alpha=16, target_modules=TARGETS, lora_dropout=0.0), 2e-4, True),
]


def quantize4(w, group=64):
    o, i = w.shape
    g = w.reshape(o, i // group, group)
    s = g.abs().amax(2, keepdim=True).clamp(min=1e-12) / 7
    return ((g / s).round().clamp(-8, 7) * s).reshape(o, i)


@torch.no_grad()
def evaluate(model, tok, data, pos, neg):
    model.eval()
    side, tok.padding_side = tok.padding_side, 'left'  # read the last real token of every row
    right = 0
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        enc = tok([prompt(s) for s, _ in chunk], return_tensors='pt', padding=True)
        logits = model(**enc).logits[:, -1]
        pred = (logits[:, pos] > logits[:, neg]).long()
        right += int((pred == torch.tensor([y for _, y in chunk])).sum())
    tok.padding_side = side
    model.train()
    return right / len(data)


def run(name, config, lr, quantized, train, val, tok):
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    if quantized:
        for n, m in base.named_modules():
            if isinstance(m, torch.nn.Linear) and '.layers.' in n:
                m.weight.data = quantize4(m.weight.data)
    total = sum(p.numel() for p in base.parameters())
    model = get_peft_model(base, config())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    pos, neg = tok.encode(' positive')[0], tok.encode(' negative')[0]
    losses, evals, start = [], [], time.perf_counter()
    model.train()
    for step in range(STEPS + 1):
        if step in EVAL_AT:
            evals.append({'step': step, 'accuracy': round(evaluate(model, tok, val, pos, neg), 4)})
            print(name, evals[-1], flush=True)
        if step == STEPS:
            break
        batch = train[step * BATCH:(step + 1) * BATCH]
        texts = [prompt(s) + (' positive' if y else ' negative') for s, y in batch]
        enc = tok(texts, return_tensors='pt', padding=True)
        labels = torch.full_like(enc.input_ids, -100)
        lengths = enc.attention_mask.sum(1)
        for r in range(len(batch)):  # right padding: the label token is the last real token
            labels[r, lengths[r] - 1] = enc.input_ids[r, lengths[r] - 1]
        out = model(**enc, labels=labels)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(round(float(out.loss), 4))
    seconds = time.perf_counter() - start
    spectrum = None
    if name in ('LoRA r=8', 'PiSSA r=8', 'DoRA r=8'):
        layer = model.base_model.model.model.layers[12].self_attn.q_proj
        delta = (layer.lora_B['default'].weight @ layer.lora_A['default'].weight) * layer.scaling['default']
        spectrum = [round(float(v), 5) for v in torch.linalg.svdvals(delta.detach())[:16]]
    return {'name': name, 'trainable': trainable, 'total': total, 'lr': lr, 'losses': losses, 'evals': evals, 'seconds': round(seconds, 1), 'spectrum': spectrum}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = 'right'
    rng = np.random.default_rng(0)
    train_all, val_all = sst2('train'), sst2('validation')
    train = [train_all[i] for i in rng.choice(len(train_all), STEPS * BATCH, replace=False)]
    val = [val_all[i] for i in rng.choice(len(val_all), EVAL_N, replace=False)]
    # Each finished variant is kept in a cache file, so an interrupted bake resumes where it stopped.
    partial = CACHE / 'lora-partial.json'
    done = {r['name']: r for r in json.loads(partial.read_text())} if partial.exists() else {}
    results = []
    for name, config, lr, quantized in VARIANTS:
        r = done.get(name) or run(name, config, lr, quantized, train, val, tok)
        results.append(r)
        partial.write_text(json.dumps(results, ensure_ascii=False))
        print(name, r['trainable'], r['evals'][-1], r['seconds'], flush=True)
    data = {'model': MODEL, 'device': 'CPU, 4 threads, float32', 'task': 'GLUE SST-2 (Socher et al. 2013): câu đánh giá phim → positive/negative',
            'steps': STEPS, 'batch': BATCH, 'evalN': EVAL_N, 'targets': TARGETS, 'examples': [[s, y] for s, y in val[:6]], 'results': results}
    path = ROOT / 'tools/kien-thuc/lora-explainer/data/lora.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(path)


if __name__ == '__main__':
    main()
