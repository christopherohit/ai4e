# ns-eval for checkpoints produced locally by our own ns-train run (trusted); torch>=2.6 defaults to weights_only=True.
import functools, sys, torch
torch.load = functools.partial(torch.load, weights_only=False)

if __name__ == '__main__':
    from nerfstudio.scripts.eval import entrypoint
    sys.exit(entrypoint())
