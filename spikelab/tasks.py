"""Synthetic tasks. Nothing is downloaded.

Contract: `Task(dt, **params)` with `n_in`, `n_classes` (None when unlabelled),
`batches(gen) -> list[(x [T, B, n_in], y [B] or None)]` for one epoch,
`test() -> (x, y)` and `metrics(net) -> dict[str, float]`.
"""
from __future__ import annotations

import torch

from .registry import register


def _counts_accuracy(net, x, y) -> float:
    with torch.no_grad():
        counts = net.run(x)["spikes"][-1].sum(0)
    return (counts.argmax(1) == y).float().mean().item()


@register("task", "patterns")
class Patterns:
    """Classify spike patterns: each class is a fixed spike-time template, seen with jitter, dropped and extra spikes."""

    defaults = {"n_in": 40, "n_classes": 4, "steps": 50, "train": 512, "test": 256, "batch": 64,
                "jitter": 2.0, "drop": 0.1, "noise_hz": 5.0, "seed": 1}

    def __init__(self, dt: float, **p):
        self.p, self.dt = {**self.defaults, **p}, dt
        self.n_in, self.n_classes = self.p["n_in"], self.p["n_classes"]
        g = torch.Generator().manual_seed(self.p["seed"])
        self.templates = torch.randint(0, self.p["steps"], (self.n_classes, self.n_in), generator=g)
        self.train = self._make(self.p["train"], g)
        self.test_set = self._make(self.p["test"], g)

    def _make(self, n, g):
        p, T = self.p, self.p["steps"]
        y = torch.randint(0, self.n_classes, (n,), generator=g)
        t = self.templates[y] + torch.round(torch.randn(n, self.n_in, generator=g) * p["jitter"]).long()
        keep = torch.rand(n, self.n_in, generator=g) >= p["drop"]
        x = torch.zeros(T, n, self.n_in)
        i, j = torch.nonzero(keep & (t >= 0) & (t < T), as_tuple=True)
        x[t[i, j], i, j] = 1.0
        x = torch.maximum(x, (torch.rand(T, n, self.n_in, generator=g) < p["noise_hz"] * self.dt).float())
        return x, y

    def batches(self, gen):
        x, y = self.train
        order = torch.randperm(x.shape[1], generator=gen)
        b = self.p["batch"]
        return [(x[:, order[k:k + b]], y[order[k:k + b]]) for k in range(0, len(order), b)]

    def test(self):
        return self.test_set

    def metrics(self, net):
        return {"test_accuracy": _counts_accuracy(net, *self.test_set)}


@register("task", "correlated")
class Correlated:
    """Poisson inputs in two halves; the first half shares a common source with correlation `c`. No labels."""

    defaults = {"n_in": 100, "rate_hz": 20.0, "c": 0.2, "steps": 1000, "batches": 20, "batch": 8}

    def __init__(self, dt: float, **p):
        self.p, self.dt = {**self.defaults, **p}, dt
        self.n_in, self.n_classes = self.p["n_in"], None
        self.half = self.n_in // 2

    def _make(self, B, gen):
        p = self.p
        r = p["rate_hz"] * self.dt
        T = p["steps"]
        src = torch.rand(T, B, 1, generator=gen) < r
        u = torch.rand(T, B, self.n_in, generator=gen)
        # Song, Miller & Abbott: a correlated input copies the shared source with probability c,
        # and fires independently at rate r(1-c) otherwise, so every input keeps rate r
        corr = (src & (u < p["c"])) | (torch.rand(T, B, self.n_in, generator=gen) < r * (1 - p["c"]))
        ind = torch.rand(T, B, self.n_in, generator=gen) < r
        x = torch.where(torch.arange(self.n_in) < self.half, corr, ind)
        return x.float()

    def batches(self, gen):
        return [(self._make(self.p["batch"], gen), None) for _ in range(self.p["batches"])]

    def test(self):
        return self._make(1, torch.Generator().manual_seed(0)), None

    def metrics(self, net):
        W = net.layers[0].W.detach()
        return {"w_correlated": W[: self.half].mean().item(), "w_uncorrelated": W[self.half :].mean().item()}
