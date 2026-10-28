# LLM bake scripts

Real measurements behind the LLM-systems labs (tools/kien-thuc/{jev,paged-attention,inference-engineering}-explainer).

```sh
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers numpy
.venv/bin/python scripts/llm-bake/jev.py        # Qwen2.5-1.5B-Instruct: autoregressive vs parallel constrained decoding
.venv/bin/python scripts/llm-bake/inference.py  # Qwen2.5-0.5B-Instruct: TTFT, decode with/without KV cache, batching; Dolly-15k lengths
```

Times are wall-clock on the machine that ran the script (recorded in each JSON as `device`); the labs say so.
Downloads go to `.cache/llm-bake` (git-ignored).
