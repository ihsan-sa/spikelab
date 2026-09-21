"""Learning rules.

Contract: `Rule(**params).fit(net, task, gen) -> history`, history a dict of lists of floats
plus "x_label" naming what one entry is (a batch, an epoch).
"""
from __future__ import annotations

import math

import torch

from .registry import ComponentError, register
from .spike import fast_sigmoid, heaviside


@register("rule", "stdp")
class STDP:
    """Pair STDP with traces, additive, weights kept in [0, w_max]. Trains the feed-forward weights online."""

    defaults = {"A_plus": 0.005, "A_minus": 0.00525, "tau_plus": 0.02, "tau_minus": 0.02, "w_max": 1.0, "epochs": 1}

    def __init__(self, **p):
        self.p = {**self.defaults, **p}

    def begin(self, net, batch: int) -> None:
        """Reset the traces; call before each simulated sequence."""
        self.net = net
        self.dp = math.exp(-net.dt / self.p["tau_plus"])
        self.dm = math.exp(-net.dt / self.p["tau_minus"])
        self.x_pre = [torch.zeros(batch, L.W.shape[0]) for L in net.layers]
        self.x_post = [torch.zeros(batch, L.W.shape[1]) for L in net.layers]

    def on_step(self, i, pre, post) -> None:
        # A pre spike in the same step as a post spike counts as causal (it was part of the input
        # that made the neuron fire), so the pre trace takes this step's spikes before the LTP term
        # and the post trace takes them after the LTD term. Window: +A_plus*exp(-d/tau_plus) for
        # d = t_post - t_pre >= 0, -A_minus*exp(d/tau_minus) for d < 0.
        p, L = self.p, self.net.layers[i]
        self.x_pre[i] = self.x_pre[i] * self.dp + pre
        self.x_post[i] = self.x_post[i] * self.dm
        B = pre.shape[0]
        dW = (p["A_plus"] * self.x_pre[i].T @ post - p["A_minus"] * pre.T @ self.x_post[i]) / B
        with torch.no_grad():
            L.W += dW * L.mask
            L.W.clamp_(0.0, p["w_max"])
        self.x_post[i] += post

    def fit(self, net, task, gen):
        hist = {"x_label": "batch"}
        with torch.no_grad():
            for _ in range(self.p["epochs"]):
                for x, _y in task.batches(gen):
                    self.begin(net, x.shape[1])
                    net.run(x, record=False, on_step=self.on_step)
                    for k, v in task.metrics(net).items():
                        hist.setdefault(k, []).append(v)
        return hist


@register("rule", "surrogate")
class SurrogateBPTT:
    """Backprop through time with a fast-sigmoid surrogate gradient (SuperSpike / SpyTorch). Loss on output spike counts."""

    defaults = {"epochs": 30, "lr": 0.002, "slope": 10.0}

    def __init__(self, **p):
        self.p = {**self.defaults, **p}

    def fit(self, net, task, gen):
        if task.n_classes is None:
            raise ComponentError(f"rule 'surrogate' needs a labelled task; task {task.name!r} has no labels")
        net.spike_fn = fast_sigmoid(self.p["slope"])
        opt = torch.optim.Adam(net.parameters(), lr=self.p["lr"])
        hist = {"x_label": "epoch"}
        try:
            for _ in range(self.p["epochs"]):
                losses = []
                for x, y in task.batches(gen):
                    counts = net.run(x)["spikes"][-1].sum(0)
                    loss = torch.nn.functional.cross_entropy(counts, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                hist.setdefault("loss", []).append(sum(losses) / len(losses))
                for k, v in task.metrics(net).items():
                    hist.setdefault(k, []).append(v)
        finally:
            net.spike_fn = heaviside
        return hist
