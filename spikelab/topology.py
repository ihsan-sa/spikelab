"""Topologies: which connections exist between two populations.

Contract: `Topology(**params).mask(n_pre, n_post, gen) -> bool tensor [n_pre, n_post]`.
"""
from __future__ import annotations

import torch

from .registry import register


@register("topology", "dense")
class Dense:
    """All-to-all."""

    defaults = {}

    def __init__(self, **p):
        pass

    def mask(self, n_pre, n_post, gen):
        return torch.ones(n_pre, n_post, dtype=torch.bool)


@register("topology", "sparse")
class Sparse:
    """Random sparse: each connection exists with probability `p`."""

    defaults = {"p": 0.2}

    def __init__(self, **p):
        self.p = {**self.defaults, **p}["p"]
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"sparse p must be in [0, 1], got {self.p}")

    def mask(self, n_pre, n_post, gen):
        return torch.rand(n_pre, n_post, generator=gen) < self.p
