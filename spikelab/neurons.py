"""Neuron models. Membrane voltage is unitless: rest 0, threshold 1 by default.

Contract (see contracts.py): `Neuron(n, dt, **params)`, `init_state(batch) -> dict`
holding at least "v" shaped [batch, n], and `step(state, I, spike_fn) -> (spikes, state)`.
`I` is the input current in voltage units (steady state v = R*I).
"""
from __future__ import annotations

import math

import torch

from .registry import register


@register("neuron", "lif")
class LIF:
    """Leaky integrate-and-fire with hard reset and an absolute refractory period."""

    defaults = {"tau_mem": 0.02, "v_th": 1.0, "v_reset": 0.0, "R": 1.0, "t_ref": 0.0}

    def __init__(self, n: int, dt: float, **p):
        self.n, self.dt, self.p = n, dt, {**self.defaults, **p}
        self.beta = math.exp(-dt / self.p["tau_mem"])  # exact decay over one step
        self.n_ref = int(round(self.p["t_ref"] / dt))

    def init_state(self, batch: int) -> dict:
        z = torch.zeros(batch, self.n)
        return {"v": z.clone(), "ref": z.clone()}

    def _integrate(self, state, I, th, spike_fn):
        p = self.p
        v = self.beta * state["v"] + (1 - self.beta) * p["R"] * I
        refr = state["ref"] > 0
        v = torch.where(refr, torch.full_like(v, p["v_reset"]), v)
        s = spike_fn(v - th)
        # reset is detached, as in SpyTorch: gradients flow through the spike, not the reset
        v = v - s.detach() * (v.detach() - p["v_reset"])
        ref = torch.clamp(state["ref"] - 1, min=0)
        ref = torch.where(s.detach() > 0, torch.full_like(ref, self.n_ref), ref)
        return s, v, ref

    def step(self, state, I, spike_fn):
        s, v, ref = self._integrate(state, I, self.p["v_th"], spike_fn)
        return s, {"v": v, "ref": ref}


@register("neuron", "adlif")
class AdaptiveLIF(LIF):
    """LIF with an adaptive threshold: each spike raises it by `b`, and it decays back with `tau_adapt`."""

    defaults = {**LIF.defaults, "tau_adapt": 0.2, "b": 0.2}

    def __init__(self, n: int, dt: float, **p):
        super().__init__(n, dt, **p)
        self.rho = math.exp(-dt / self.p["tau_adapt"])

    def init_state(self, batch: int) -> dict:
        return {**super().init_state(batch), "a": torch.zeros(batch, self.n)}

    def step(self, state, I, spike_fn):
        th = self.p["v_th"] + self.p["b"] * state["a"]
        s, v, ref = self._integrate(state, I, th, spike_fn)
        a = self.rho * state["a"] + s
        return s, {"v": v, "ref": ref, "a": a}
