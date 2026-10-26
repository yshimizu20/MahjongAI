"""The trainer's opt-in --compile (T-050 Part 2): one torch.compile(dynamic=False) callable per head for the network
after the token embedder (EncoderConvModel.head_network: state net, fusion, encoder blocks, ln_f, head, mask fill,
cross entropy). The token embedder (torch.unique: a data-dependent size) stays eager in forward_heads, so every
compiled input has a static shape: one graph per (head, B) signature.

The callables live here, outside the model's module tree: the model's state_dict keys stay as without --compile (no
`_orig_mod.`), a deepcopy of the model (the trainer's CPU eval copy) copies no compiled code, and evaluation (which
calls the model, never forward_heads with cores) stays eager. Inductor draws dropout from the eager RNG
(fallback_random), but the draws are not those of the eager path on MPS (other draw count and order): compiled
training is not bit-reproducible against eager with dropout on; its dropout distribution is (test_compile.py)."""
import torch
import torch.nn as nn

COMPILE_ON_LINE = "compile on | torch.compile per head, dynamic=False, fallback_random True"
_fallbacks = []  # every op Inductor left to an aten fallback kernel in this process, by name (install_fallback_spy)


def install_fallback_spy():
    """Record the op of every Inductor FallbackKernel this process creates (G2's spy; once per process). It fires
    when Inductor lowers a graph: a graph loaded whole from the Inductor cache records nothing."""
    import torch._inductor.ir as ir
    orig = ir.FallbackKernel.__dict__["create"].__func__
    if getattr(orig, "_t050_spy", False):
        return

    def spy(cls, kernel, *args, **kwargs):
        _fallbacks.append(str(kernel))
        return orig(cls, kernel, *args, **kwargs)

    spy._t050_spy = True
    ir.FallbackKernel.create = classmethod(spy)


def fallback_ops() -> list:
    """The distinct fallback op names recorded so far in this process, sorted."""
    return sorted(set(_fallbacks))


class HeadCore(nn.Module):
    """One head's compiled region: model.head_network(head, ...). Holds the model as a submodule of its own; the model
    does not hold it."""

    def __init__(self, model, head: str):
        super().__init__()
        self.model, self.head = model, head

    def forward(self, history, state, padding, mask, y):
        return self.model.head_network(self.head, history, state, padding, mask, y)


def compile_core(core: nn.Module):
    """The production compiler of one HeadCore (CPU tests replace it with a counting stub)."""
    return torch.compile(core, dynamic=False)


class CompiledHeads:
    """{head: compiled HeadCore} of one model, called as model.head_network is: (head, history, state, padding, mask,
    y) -> (logits, loss). Records the (head, B) signature of each call. core: the HeadCore class (a test passes a
    subclass: a tapped or a broken region)."""

    def __init__(self, model, heads=("during", "discard", "post"), core=HeadCore):
        import torch._inductor.config as inductor_config
        inductor_config.fallback_random = True
        install_fallback_spy()
        self.cores = {h: compile_core(core(model, h)) for h in heads}
        self.signatures = set()

    def __call__(self, head: str, history, state, padding, mask, y):
        self.signatures.add((head, history.size(0)))
        return self.cores[head](history, state, padding, mask, y)

    def summary(self) -> str:
        """The run's one end line (the G3 judge parses it): dynamo graphs of this process, distinct (head, B)
        signatures called, graph breaks, the fallback ops recorded by the spy, and the Inductor / AOT autograd cache
        hits. The spy sees only graphs lowered in this process: the fallback list is complete only when both hit
        counts are 0 (a cold cache); a judge gates fallbacks on such a run."""
        from torch._dynamo.utils import counters
        return (f"compile summary | graphs {counters['stats']['unique_graphs']} | signatures {len(self.signatures)} | "
                f"graph breaks {sum(counters['graph_break'].values())} | fallbacks [{', '.join(fallback_ops())}] | "
                f"cache hits fx {counters['inductor']['fxgraph_cache_hit']} "
                f"aot {counters['aot_autograd']['autograd_cache_hit']}")
