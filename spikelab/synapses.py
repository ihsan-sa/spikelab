"""Synapse models: turn weighted presynaptic spikes into a current.

Contract: `Synapse(n, dt, **params)`, `init_state(batch) -> dict`, `step(state, x) -> (I, state)`
with x the weighted spike input shaped [batch, n]. Both built-ins deliver the same total
charge per spike (sum of I over time equals x), so swapping them changes the shape, not the size.
"""
from __future__ import annotations

import math

import torch

from .registry import register


@register("synapse", "delta")
class Delta:
    """Delta current: the whole input arrives in the same step."""

    defaults = {}

    def __init__(self, n: int, dt: float, **p):
        self.n = n

    def init_state(self, batch: int) -> dict:
        return {}

    def step(self, state, x):
        return x, state


@register("synapse", "exponential")
class Exponential:
    """Exponential current: each spike starts a current that decays with `tau_syn`."""

    defaults = {"tau_syn": 0.005}

    def __init__(self, n: int, dt: float, **p):
        self.n, self.p = n, {**self.defaults, **p}
        self.alpha = math.exp(-dt / self.p["tau_syn"])

    def init_state(self, batch: int) -> dict:
        return {"I": torch.zeros(batch, self.n)}

    def step(self, state, x):
        I = self.alpha * state["I"] + (1 - self.alpha) * x
        return I, {"I": I}
