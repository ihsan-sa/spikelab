"""Known results the engine must reproduce. Used by the tests and by scripts/make_figures.py."""
from __future__ import annotations


import numpy as np
import torch

from . import config, registry
from .network import Layer, Network
from .spike import heaviside


def lif_rate_theory(I, tau=0.02, t_ref=0.002, R=1.0, v_th=1.0):
    """Analytic LIF rate for constant current: 1 / (t_ref + tau ln(RI / (RI - v_th))), 0 below rheobase."""
    RI = R * np.asarray(I, dtype=float)
    out = np.zeros_like(RI)
    ok = RI > v_th
    out[ok] = 1.0 / (t_ref + tau * np.log(RI[ok] / (RI[ok] - v_th)))
    return out


def lif_rate_sim(I, tau=0.02, t_ref=0.002, dt=1e-4, seconds=1.0):
    """Drive one LIF per current (batch = currents) and measure rate from the mean inter-spike interval."""
    registry.load_builtins()
    lif = registry.get("neuron", "lif")(1, dt, tau_mem=tau, t_ref=t_ref)
    I = torch.tensor(np.asarray(I, dtype=np.float32)).reshape(-1, 1)
    st = lif.init_state(len(I))
    times = [[] for _ in range(len(I))]
    for k in range(int(seconds / dt)):
        s, st = lif.step(st, I, heaviside)
        for b in torch.nonzero(s[:, 0]).flatten().tolist():
            times[b].append(k * dt)
    return np.array([(len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 else 0.0 for t in times])


def stdp_window(deltas_ms, dt=1e-3, **params):
    """Weight change for a single pre/post pair at each delta = t_post - t_pre, using the real STDP rule."""
    registry.load_builtins()
    rule = registry.get("rule", "stdp")(**params)
    out = []
    for d in deltas_ms:
        k = int(round(d * 1e-3 / dt))
        W = torch.nn.Parameter(torch.full((1, 1), 0.5))
        net = Network([Layer(W=W, mask=torch.ones(1, 1, dtype=torch.bool), neuron=None, synapse=None)], dt)
        rule.begin(net, 1)
        t_pre = 100
        for t in range(2 * t_pre + 1):
            pre = torch.tensor([[float(t == t_pre)]])
            post = torch.tensor([[float(t == t_pre + k)]])
            rule.on_step(0, pre, post)
        out.append(W.item() - 0.5)
    return np.array(out)


def stdp_theory(deltas_ms, A_plus, A_minus, tau_plus, tau_minus):
    d = np.asarray(deltas_ms, dtype=float) * 1e-3
    return np.where(d >= 0, A_plus * np.exp(-d / tau_plus), -A_minus * np.exp(d / tau_minus))


def run_config(path, **overrides):
    """Build and train a config, returning (net, task, history). overrides: {section: {key: value}}."""
    cfg = config.load(path)
    for sec, kv in overrides.items():
        cfg.setdefault(sec, {}).update(kv)
    cfg, net, rule, task, gen = config.build(cfg)
    return net, task, rule.fit(net, task, gen)


def bimodality(w, w_max):
    """Fraction of weights within 10% of either bound."""
    w = np.asarray(w)
    return float(((w < 0.1 * w_max) | (w > 0.9 * w_max)).mean())


__all__ = ["lif_rate_theory", "lif_rate_sim", "stdp_window", "stdp_theory", "run_config", "bimodality"]
