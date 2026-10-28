"""Measurements for the Inference Engineering and PagedAttention labs.

    .venv/bin/python scripts/llm-bake/inference.py

Model: Qwen/Qwen2.5-0.5B-Instruct (24 layers, hidden 896, 14 query heads / 2 KV heads, head dim 64), float32 on a
4-thread CPU. Absolute times are for this CPU only; the lab uses them to show shapes (prefill vs decode, with vs without
KV cache, batching) and says so.

Workload lengths: databricks-dolly-15k (CC BY-SA 3.0), 15 011 human-written instruction/response pairs; prompt =
instruction + context, output = response, counted with the Qwen2.5 tokenizer.
"""
import json, time, urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/llm-bake'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
DOLLY = 'https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl'
torch.set_num_threads(4)
torch.manual_seed(0)


def timed(fn, repeat=3):
    best = float('inf')
    for _ in range(repeat):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best


@torch.no_grad()
def measure(model, tok):
    vocab = model.config.vocab_size
    ids = lambda n, b=1: torch.randint(1000, 20000, (b, n))
    # 1. Prefill time vs prompt length (time to first token).
    prefill = []
    for n in [16, 64, 128, 256, 512, 1024, 2048]:
        x = ids(n)
        prefill.append({'tokens': n, 'seconds': round(timed(lambda: model(x, use_cache=True)), 4)})
    # 2. Decode 96 tokens after a 128-token prompt, with and without the KV cache.
    prompt = ids(128)
    out = model(prompt, use_cache=True)
    cache, token, with_cache = out.past_key_values, out.logits[:, -1:].argmax(-1), []
    for _ in range(96):
        t = time.perf_counter(); out = model(token, past_key_values=cache, use_cache=True); with_cache.append(time.perf_counter() - t)
        cache, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
    seq, without = prompt.clone(), []
    for _ in range(96):
        t = time.perf_counter(); o = model(seq, use_cache=False); without.append(time.perf_counter() - t)
        seq = torch.cat([seq, o.logits[:, -1:].argmax(-1)], 1)
    # 3. Batched decode: one step for B sequences of 256 cached tokens.
    batch = []
    for b in [1, 2, 4, 8, 16, 32, 64]:
        o = model(ids(256, b), use_cache=True)
        c, tkn = o.past_key_values, o.logits[:, -1:].argmax(-1)
        steps = []
        for _ in range(8):
            t = time.perf_counter(); o = model(tkn, past_key_values=c, use_cache=True); steps.append(time.perf_counter() - t)
            c, tkn = o.past_key_values, o.logits[:, -1:].argmax(-1)
        step = float(np.median(steps))
        batch.append({'batch': b, 'stepSeconds': round(step, 4), 'tokensPerSecond': round(b / step, 1)})
    # 4. KV cache size per token, read from the real cache tensors.
    o = model(ids(100), use_cache=True)
    layers = o.past_key_values.layers if hasattr(o.past_key_values, 'layers') else None
    if layers is not None:
        total = sum(l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size() for l in layers)
    else:
        total = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in o.past_key_values)
    return {
        'prefill': prefill,
        'decodeWithCache': [round(v, 4) for v in with_cache],
        'decodeWithoutCache': [round(v, 4) for v in without],
        'batch': batch,
        'kvBytesPerTokenFp32': total // 100,
        'vocab': vocab,
    }


def dolly(tok):
    path = CACHE / 'dolly.jsonl'
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DOLLY, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    prompt = np.array([len(tok.encode(r['instruction'] + '\n' + r['context'])) for r in rows])
    output = np.array([len(tok.encode(r['response'])) for r in rows])
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), 600, replace=False)
    q = lambda a: {str(p): int(np.percentile(a, p)) for p in [10, 25, 50, 75, 90, 99]}
    return {
        'source': 'databricks-dolly-15k (Databricks 2023, CC BY-SA 3.0): 15 011 cặp chỉ dẫn/câu trả lời do người viết; đếm token bằng tokenizer Qwen2.5.',
        'count': len(rows), 'prompt': q(prompt), 'output': q(output), 'promptMean': round(float(prompt.mean()), 1), 'outputMean': round(float(output.mean()), 1),
        'sample': [[int(prompt[i]), int(output[i])] for i in pick],
    }


# KV cache per token in fp16 for reference models: 2 (K and V) × layers × KV heads × head dim × 2 bytes.
REFERENCE = [
    {'name': 'Qwen2.5-0.5B', 'layers': 24, 'kvHeads': 2, 'headDim': 64},
    {'name': 'Qwen2.5-1.5B', 'layers': 28, 'kvHeads': 2, 'headDim': 128},
    {'name': 'Qwen2.5-7B', 'layers': 28, 'kvHeads': 4, 'headDim': 128},
    {'name': 'Llama-2-13B', 'layers': 40, 'kvHeads': 40, 'headDim': 128},
    {'name': 'Llama-3-70B', 'layers': 80, 'kvHeads': 8, 'headDim': 128},
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    model(torch.randint(1000, 2000, (1, 32)))  # warm-up
    c = model.config
    data = {'model': MODEL, 'device': 'CPU, 4 threads, float32', 'layers': c.num_hidden_layers, 'hidden': c.hidden_size, 'heads': c.num_attention_heads,
            'kvHeads': c.num_key_value_heads, 'params': sum(p.numel() for p in model.parameters()), **measure(model, tok)}
    data['reference'] = [{**r, 'kvBytesPerTokenFp16': 2 * r['layers'] * r['kvHeads'] * r['headDim'] * 2} for r in REFERENCE]
    data['workload'] = dolly(tok)
    for name in ['inference-engineering-explainer', 'paged-attention-explainer']:
        path = ROOT / f'tools/kien-thuc/{name}/data/inference.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
        print(path, path.stat().st_size // 1024, 'KB')
    print(json.dumps({k: v for k, v in data.items() if k not in ('decodeWithCache', 'decodeWithoutCache', 'workload')}, indent=1)[:2500])
    print(data['workload']['prompt'], data['workload']['output'])


if __name__ == '__main__':
    main()
