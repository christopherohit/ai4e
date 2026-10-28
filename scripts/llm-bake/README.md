# LLM bake scripts

Real measurements behind the LLM-systems labs (tools/kien-thuc/{jev,paged-attention,inference-engineering,moe,quantization,lora}-explainer).

```sh
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers numpy scikit-learn peft pyarrow
.venv/bin/python scripts/llm-bake/jev.py        # Qwen2.5-1.5B-Instruct: autoregressive vs parallel constrained decoding
.venv/bin/python scripts/llm-bake/inference.py  # Qwen2.5-0.5B-Instruct: TTFT, decode with/without KV cache, batching; Dolly-15k lengths
.venv/bin/python scripts/llm-bake/moe.py        # small MoE on UCI digits: router, top-2, load-balancing loss (teaching model)
.venv/bin/python scripts/llm-bake/quantize.py   # Qwen2.5-0.5B-Instruct: RTN INT8/4/3/2, per-tensor/channel/group, WikiText-2 perplexity
.venv/bin/python scripts/llm-bake/lora.py       # Qwen2.5-0.5B-Instruct on SST-2: LoRA, rsLoRA, DoRA, PiSSA, VeRA, LoHa, QLoRA-style (resumable)
```

Times are wall-clock on the machine that ran the script (recorded in each JSON as `device`); the labs say so.
Downloads go to `.cache/llm-bake` (git-ignored).
