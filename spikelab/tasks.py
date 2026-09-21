"""Synthetic tasks. Nothing is downloaded.

Contract: `Task(dt, **params)` with `n_in`, `n_classes` (None when unlabelled),
`batches(gen) -> list[(x [T, B, n_in], y [B] or None)]` for one epoch,
`test() -> (x, y)` and `metrics(net) -> dict[str, float]`.
x is spikes, unless the task sets `drive = "current"`: then x is a current and the network's first
layer is real neurons driven by it (network.drive).
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


@register("task", "current")
class Current:
    """Inject a current straight into the input neurons: constant, step, ramp or noisy. No labels."""

    defaults = {"n_in": 10, "waveform": "constant", "amplitude": 1.5, "onset": 0.1, "offset": 0.4,
                "sigma": 0.3, "spread": 0.0, "steps": 500, "batches": 4, "batch": 1, "seed": 1}
    drive = "current"  # config.build gives the network a layer of input neurons that this current drives
    WAVEFORMS = ("constant", "step", "ramp", "noisy")

    def __init__(self, dt: float, **p):
        self.p, self.dt = {**self.defaults, **p}, dt
        self.n_in, self.n_classes = self.p["n_in"], None
        if self.p["waveform"] not in self.WAVEFORMS:
            raise ValueError(f"waveform must be one of {', '.join(self.WAVEFORMS)}; got {self.p['waveform']!r}")
        self.test_set = self._make(1, torch.Generator().manual_seed(self.p["seed"]))

    def _make(self, B, gen):
        """Current [T, B, n_in] in voltage units (steady state v = R*I)."""
        p = self.p
        t = torch.arange(p["steps"]).float() * self.dt
        on, off = p["onset"], p["offset"]
        wave = {
            "constant": torch.ones_like(t),
            "step": ((t >= on) & (t < off)).float(),  # amplitude from onset until offset, 0 otherwise
            "ramp": ((t - on) / max(off - on, self.dt)).clamp(0.0, 1.0),  # 0 before onset, full from offset on
            "noisy": torch.ones_like(t),
        }[p["waveform"]]
        # spread: neuron k gets amplitude * (1 - spread ... 1 + spread), evenly, so neurons differ
        scale = 1.0 + p["spread"] * torch.linspace(-1.0, 1.0, self.n_in) if self.n_in > 1 else torch.ones(1)
        I = p["amplitude"] * wave[:, None, None] * scale[None, None, :]
        I = I.expand(-1, B, -1).clone()
        if p["waveform"] == "noisy":
            I += p["sigma"] * torch.randn(I.shape, generator=gen)
        return I

    def batches(self, gen):
        return [(self._make(self.p["batch"], gen), None) for _ in range(self.p["batches"])]

    def test(self):
        return self.test_set, None

    def metrics(self, net):
        with torch.no_grad():
            rec = net.run(self.test_set)
        T = self.test_set.shape[0] * self.dt
        return {"input_rate_hz": rec["input_spikes"].mean().item() * self.test_set.shape[0] / T,
                "output_rate_hz": rec["spikes"][-1].mean().item() * self.test_set.shape[0] / T}
