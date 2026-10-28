"""Measure "Jev-style" parallel constrained decoding against ordinary autoregressive JSON generation.

    python -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv/bin/pip install transformers
    .venv/bin/python scripts/llm-bake/jev.py

Model: Qwen/Qwen2.5-1.5B-Instruct (28 layers, hidden size 1536, vocabulary 151 936), float32 on CPU.
The three documents are short fictional incident reports written for this lab (no real people or companies).

For each document:
  A. autoregressive: greedy generation of the whole JSON object, token by token (KV cache on);
  B. parallel: one prefill of prompt + schema + '{', then ONE batched forward pass that appends every field's suffix
     plus every candidate value (right-padded, sharing the prefilled cache) and sums the log-probabilities of each
     candidate's tokens. Multi-token values (MEDIUM, TIER_1…) are scored on all their tokens.
Wall-clock times are for this CPU (torch threads = 4) and only meant for comparison between A and B.
"""
import json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
torch.set_num_threads(4)
torch.manual_seed(0)

SCHEMA = {
    'risk_level': ['HIGH', 'MEDIUM', 'LOW', 'NONE'],
    'requires_review': ['true', 'false'],
    'action_tier': ['TIER_1', 'TIER_2', 'TIER_3'],
}
QUOTED = {'risk_level': True, 'requires_review': False, 'action_tier': True}
SCHEMA_TEXT = '{"risk_level": HIGH | MEDIUM | LOW | NONE, "requires_review": true | false, "action_tier": TIER_1 | TIER_2 | TIER_3}'
RULES = ('risk_level: how severe the incident is. requires_review: true if a human must check it. '
         'action_tier: TIER_1 = routine fix, TIER_2 = escalate to the team lead, TIER_3 = emergency response.')

DOCS = [
    ('Rò rỉ dữ liệu', 'Incident report #A-17. At 02:14 the monitoring system detected that a misconfigured storage bucket had been publicly readable for 36 hours. '
     'It contained exported customer records including names, emails and partial card numbers. Access logs show downloads from three unknown IP addresses. '
     'The bucket has now been locked, but the data may already be circulating.'),
    ('Máy in hỏng', 'Ticket #B-203. The shared printer on the third floor jams every few pages since Monday. Staff can still print on the second floor. '
     'No data or security impact. The vendor can replace the roller during the next scheduled visit.'),
    ('Đăng nhập lạ', 'Alert #C-58. A staff account logged in from a new country at 23:40, then changed its recovery email. The user says it was not them. '
     'No files were downloaded and the session was revoked after 12 minutes. Password reset is pending.'),
]


def prompt(tok, doc):
    messages = [
        {'role': 'system', 'content': f'You extract structured fields from documents. Answer only with JSON following this schema: {SCHEMA_TEXT}. {RULES}'},
        {'role': 'user', 'content': doc},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '{'


def suffix(field, first):
    # Quoted values start right after the opening quote; bare values (true/false) are tokenised with their leading
    # space (" true" is one token), so the suffix stops at the colon and the candidate carries the space.
    return ('' if first else ' ') + f'"{field}":' + (' "' if QUOTED[field] else '')


def candidate(field, value):
    return value if QUOTED[field] else ' ' + value


@torch.no_grad()
def autoregressive(model, tok, text):
    ids = tok(text, return_tensors='pt').input_ids
    start = time.perf_counter()
    out = model(ids, use_cache=True)
    first = time.perf_counter() - start
    cache, nxt, generated, steps = out.past_key_values, out.logits[0, -1].argmax(), [], [first]
    for _ in range(80):
        token = int(nxt)
        generated.append(token)
        if '}' in tok.decode([token]) or token == tok.eos_token_id:
            break
        t0 = time.perf_counter()
        out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        steps.append(time.perf_counter() - t0)
        cache, nxt = out.past_key_values, out.logits[0, -1].argmax()
    total = time.perf_counter() - start
    body = '{' + tok.decode(generated)
    try:
        parsed = json.loads(body[: body.index('}') + 1])
        valid = all(parsed.get(k) in v or str(parsed.get(k)).lower() in v for k, v in SCHEMA.items())
    except (ValueError, json.JSONDecodeError):
        parsed, valid = None, False
    return {'tokens': [tok.decode([t]) for t in generated], 'text': body, 'forwardPasses': len(steps), 'seconds': round(total, 3),
            'ttft': round(first, 3), 'perToken': round(sum(steps[1:]) / max(1, len(steps) - 1), 4), 'parsed': parsed, 'valid': valid}


@torch.no_grad()
def parallel(model, tok, text):
    ids = tok(text, return_tensors='pt').input_ids
    t0 = time.perf_counter()
    pre = model(ids, use_cache=True)
    prefill = time.perf_counter() - t0
    # Every (field, candidate) row: field suffix + candidate tokens; we need log P(candidate | prefix + suffix).
    rows, meta = [], []
    for i, (field, cands) in enumerate(SCHEMA.items()):
        s = tok.encode(suffix(field, i == 0))
        for c in cands:
            ct = tok.encode(candidate(field, c))
            rows.append(s + ct)
            meta.append((field, c, len(s), ct))
    width = max(len(r) for r in rows)
    batch = torch.full((len(rows), width), tok.pad_token_id or 0)
    for r, row in enumerate(rows):
        batch[r, : len(row)] = torch.tensor(row)
    cache = pre.past_key_values
    cache.batch_repeat_interleave(len(rows))
    t1 = time.perf_counter()
    out = model(batch, past_key_values=cache, use_cache=False, output_hidden_states=True)
    score = time.perf_counter() - t1
    logp = torch.log_softmax(out.logits.float(), -1)
    last_prefix = torch.log_softmax(pre.logits[0, -1].float(), -1)
    fields = {}
    for r, (field, cand, slen, ct) in enumerate(meta):
        total = 0.0
        for j, token in enumerate(ct):
            pos = slen + j - 1  # logits at position p predict token p + 1
            total += float(logp[r, pos, token])
        f = fields.setdefault(field, {'candidates': [], 'suffix': suffix(field, field == 'risk_level'), 'suffixTokens': [tok.decode([t]) for t in tok.encode(suffix(field, field == 'risk_level'))]})
        f['candidates'].append({'value': cand, 'tokens': [tok.decode([t]) for t in ct], 'logprob': round(total, 4)})
        if 'firstRow' not in f:
            f['firstRow'] = r
            f['slen'] = slen
    for field, f in fields.items():
        lp = torch.tensor([c['logprob'] for c in f['candidates']])
        probs = torch.softmax(lp, 0)
        for c, p in zip(f['candidates'], probs):
            c['p'] = round(float(p), 5)
        r, pos = f.pop('firstRow'), f.pop('slen') - 1
        full = logp[r, pos]
        top = torch.topk(full, 6)
        f['top'] = [{'token': tok.decode([int(i)]), 'id': int(i), 'p': round(float(v.exp()), 5), 'logit': round(float(out.logits[r, pos, int(i)]), 3)} for v, i in zip(top.values, top.indices)]
        raw = out.logits[r, pos].float()
        f['firstTokenLogits'] = [{'value': c['value'], 'token': c['tokens'][0], 'id': tok.encode(candidate(field, c['value']))[0], 'logit': round(float(raw[tok.encode(candidate(field, c['value']))[0]]), 3)} for c in f['candidates']]
        f['vocabLogitStats'] = {'max': round(float(raw.max()), 3), 'mean': round(float(raw.mean()), 3), 'std': round(float(raw.std()), 3)}
        hidden = out.hidden_states[-1][r, pos].float()
        f['hidden'] = [round(float(v), 2) for v in hidden]
        f['answer'] = max(f['candidates'], key=lambda c: c['p'])['value']
    return {'promptTokens': int(ids.shape[1]), 'prefillSeconds': round(prefill, 3), 'scoreSeconds': round(score, 3), 'rows': len(rows),
            'seconds': round(prefill + score, 3), 'forwardPasses': 2, 'fields': fields, 'nextAfterPrefix': [{'token': tok.decode([int(i)]), 'p': round(float(v.exp()), 4)} for v, i in zip(*torch.topk(last_prefix, 5))]}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    c = model.config
    head = c.hidden_size // c.num_attention_heads
    out = {'model': MODEL, 'device': 'CPU, 4 threads, float32', 'layers': c.num_hidden_layers, 'hidden': c.hidden_size, 'vocab': c.vocab_size,
           'kvHeads': c.num_key_value_heads, 'heads': c.num_attention_heads, 'headDim': head, 'schema': SCHEMA, 'schemaText': SCHEMA_TEXT, 'rules': RULES, 'docs': []}
    # Warm-up so the first timed run does not pay one-off costs.
    autoregressive(model, tok, prompt(tok, DOCS[1][1]))
    for title, doc in DOCS:
        text = prompt(tok, doc)
        ar, par = autoregressive(model, tok, text), parallel(model, tok, text)
        out['docs'].append({'title': title, 'text': doc, 'autoregressive': ar, 'parallel': par})
        print(title, 'AR', ar['forwardPasses'], ar['seconds'], ar['valid'], ar['text'], '| PAR', par['seconds'], {k: v['answer'] for k, v in par['fields'].items()})
    path = ROOT / 'tools/kien-thuc/jev-explainer/data/jev.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(path, path.stat().st_size // 1024, 'KB')


if __name__ == '__main__':
    main()
