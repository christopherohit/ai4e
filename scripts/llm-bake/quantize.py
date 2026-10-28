"""Weight-only quantization of Qwen2.5-0.5B-Instruct, measured on CPU.

    .venv/bin/python scripts/llm-bake/quantize.py

Round-to-nearest (RTN) symmetric integer quantization of every Linear weight in the transformer blocks (embeddings and
LM head stay in float32), simulated by quantize → dequantize so the maths of each format is exact:
  per-tensor  one scale per matrix
  per-channel one scale per output row
  group-g     one scale per g consecutive inputs of a row (as used by GPTQ / AWQ / llama.cpp k-quants)
Quality: perplexity on the first 16 384 tokens of WikiText-2 (raw, test split), 512-token windows, and a greedy
continuation of a fixed Vietnamese prompt. Also stores one real weight row and per-layer weight statistics.
"""
import io, json, math, urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/llm-bake'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
WIKI = 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet'
torch.set_num_threads(4)

CONFIGS = [
    ('fp32', None, None), ('int8 per-tensor', 8, 'tensor'), ('int8 per-channel', 8, 'channel'),
    ('int4 per-tensor', 4, 'tensor'), ('int4 per-channel', 4, 'channel'), ('int4 group-128', 4, 128), ('int4 group-32', 4, 32),
    ('int3 group-128', 3, 128), ('int2 group-32', 2, 32),
]
PROMPT = 'Thủ đô của Việt Nam là thành phố nào? Trả lời bằng một câu.'


def quantize(w, bits, gran):
    qmax = 2 ** (bits - 1) - 1
    if gran == 'tensor':
        s = w.abs().max() / qmax
        return (w / s).round().clamp(-qmax - 1, qmax) * s
    if gran == 'channel':
        s = w.abs().amax(1, keepdim=True) / qmax
        return (w / s).round().clamp(-qmax - 1, qmax) * s
    o, i = w.shape
    g = w.reshape(o, i // gran, gran)
    s = g.abs().amax(2, keepdim=True).clamp(min=1e-12) / qmax
    return ((g / s).round().clamp(-qmax - 1, qmax) * s).reshape(o, i)


def bits_per_weight(bits, gran):
    if bits is None: return 32
    if gran in ('tensor', 'channel'): return bits  # scales are negligible
    return bits + 16 / gran  # one fp16 scale per group


def wikitext(tok):
    path = CACHE / 'wikitext2-test.parquet'
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WIKI, path)
    import pyarrow.parquet as pq
    text = '\n\n'.join(pq.read_table(path).column('text').to_pylist())
    return tok(text, return_tensors='pt').input_ids[0, :16384]


@torch.no_grad()
def perplexity(model, ids, window=512):
    nll, count = 0.0, 0
    for i in range(0, len(ids) - 1, window):
        x = ids[i:i + window + 1].unsqueeze(0)
        if x.shape[1] < 2: break
        out = model(x[:, :-1])
        nll += torch.nn.functional.cross_entropy(out.logits[0], x[0, 1:], reduction='sum').item()
        count += x.shape[1] - 1
    return math.exp(nll / count)


@torch.no_grad()
def continuation(model, tok):
    text = tok.apply_chat_template([{'role': 'user', 'content': PROMPT}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors='pt').input_ids
    out = model.generate(ids, max_new_tokens=28, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and '.layers.' in n}
    original = {n: m.weight.data.clone() for n, m in linears.items()}
    ids = wikitext(tok)
    sample_name = 'model.layers.12.mlp.down_proj'
    row = original[sample_name][7].numpy()
    stats = []
    for n, w in original.items():
        a = w.abs()
        stats.append({'name': n.replace('model.layers.', 'L'), 'absMax': round(float(a.max()), 4), 'std': round(float(w.std()), 5), 'kurtosis': round(float(((w - w.mean()) ** 4).mean() / w.var() ** 2), 2)})
    params = sum(w.numel() for w in original.values())
    results = []
    for name, bits, gran in CONFIGS:
        err = 0.0
        for n, m in linears.items():
            w = original[n]
            m.weight.data = w if bits is None else quantize(w, bits, gran)
            err += float(((m.weight.data - w) ** 2).sum())
        ppl = perplexity(model, ids)
        text = continuation(model, tok)
        r = {'name': name, 'bits': bits, 'granularity': gran, 'bitsPerWeight': bits_per_weight(bits, gran), 'perplexity': round(ppl, 3),
             'weightRmse': round(math.sqrt(err / params), 6), 'continuation': text}
        print(r)
        results.append(r)
    for n, m in linears.items():
        m.weight.data = original[n]
    # One real weight row quantized at several settings, for the lab's histogram chapter.
    w = torch.tensor(row).unsqueeze(0)
    rows = {f'{b}-{g}': quantize(w, b, g)[0, :512].numpy().round(5).tolist() for b in (8, 4, 3, 2) for g in ('tensor', 128, 32)}
    data = {
        'model': MODEL, 'device': 'CPU, float32 (quantize → dequantize)', 'quantizedParams': params, 'totalParams': sum(p.numel() for p in model.parameters()),
        'evalTokens': int(len(ids)), 'evalSource': 'WikiText-2 raw, test split (Merity et al. 2016, CC BY-SA 3.0)', 'prompt': PROMPT,
        'results': results, 'layerStats': stats, 'sampleName': sample_name, 'sampleRowLength': int(len(row)), 'sampleRowAbsMax': round(float(abs(row).max()), 5), 'sampleRow': row[:512].round(5).tolist(), 'sampleQuantized': rows,
    }
    path = ROOT / 'tools/kien-thuc/quantization-explainer/data/quantization.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(path, path.stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
