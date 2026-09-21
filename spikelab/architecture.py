"""Architectures: how many layers, how big, recurrent or not, and the starting weights.

Contract: `Architecture(**params).build(neuron, synapse, topology, dt, gen) -> Network`, where
neuron and synapse are factories `f(n) -> instance` and topology is an instance.
"""
from __future__ import annotations

import torch

from .network import Layer, Network
from .registry import register


@register("architecture", "layered")
class Layered:
    """Feed-forward layers `sizes` = [inputs, hidden..., outputs]; `recurrent` adds recurrent weights to hidden layers."""

    defaults = {"sizes": [20, 10], "recurrent": False, "init": "normal", "weight_scale": 1.0}

    def __init__(self, **p):
        self.p = {**self.defaults, **p}
        if len(self.p["sizes"]) < 2 or min(self.p["sizes"]) < 1:
            raise ValueError(f"sizes needs at least [inputs, outputs], all >= 1; got {self.p['sizes']}")
        if self.p["init"] not in ("normal", "uniform"):
            raise ValueError(f"init must be 'normal' or 'uniform', got {self.p['init']!r}")

    def _weights(self, n_pre, n_post, mask, gen):
        scale = self.p["weight_scale"]
        if self.p["init"] == "normal":  # zero mean, variance scaled by fan-in (for gradient training)
            w = torch.randn(n_pre, n_post, generator=gen) * scale / n_pre**0.5
        else:  # uniform in [0, weight_scale] (for STDP, which keeps weights in [0, w_max])
            w = torch.rand(n_pre, n_post, generator=gen) * scale
        return torch.nn.Parameter(w * mask)

    def build(self, neuron, synapse, topology, dt, gen) -> Network:
        sizes, layers = self.p["sizes"], []
        for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            mask = topology.mask(a, b, gen)
            L = Layer(W=self._weights(a, b, mask, gen), mask=mask, neuron=neuron(b), synapse=synapse(b))
            hidden = k < len(sizes) - 2
            if self.p["recurrent"] and hidden:
                vmask = topology.mask(b, b, gen) & ~torch.eye(b, dtype=torch.bool)
                L.V, L.vmask = self._weights(b, b, vmask, gen), vmask
            layers.append(L)
        return Network(layers, dt)
