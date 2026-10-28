# sparse-attention-explainer

Glass lab (tools/_shared/glass-lab) for blog/src/content/posts/kien-thuc/xu-ly-ngon-ngu/sparse-attention.mdx.
data/attention.json: real attention of Qwen2.5-0.5B-Instruct on 1 024 WikiText-2 tokens, baked by
scripts/llm-bake/sparse_attention.py (coverage of local / sink + local / block top-k / oracle per layer and head,
four pooled maps and query rows). model.ts also keeps the 32-token code example used for QSA's three steps.
