"""The contract each kind of component must meet, checked when it registers.

Each check is a small smoke run on stand-ins, so a component is judged on its own:
a broken LIF cannot make the architecture check fail.
"""
from __future__ import annotations

import torch

from .registry import contracts
from .spike import heaviside

DT = 1e-3


class _StubNeuron:
    def __init__(self, n):
        self.n = n

    def init_state(self, batch):
        return {"v": torch.zeros(batch, self.n)}

    def step(self, state, I, spike_fn):
        v = state["v"] + I
        s = spike_fn(v - 1.0)
        return s, {"v": v * (1 - s)}


class _StubSynapse:
    def init_state(self, batch):
        return {}

    def step(self, state, x):
        return x, state


class _StubTopology:
    def mask(self, a, b, gen):
        return torch.ones(a, b, dtype=torch.bool)


def _check_spikes(s, shape):
    assert s.shape == shape, f"spikes shaped {tuple(s.shape)}, expected {shape}"
    assert bool(((s == 0) | (s == 1)).all()), "spikes must be 0 or 1"


def neuron(cls):
    nrn = cls(3, DT, **cls.defaults)
    for drive, want in ((0.0, False), (5.0, True)):
        st = nrn.init_state(2)
        assert "v" in st and st["v"].shape == (2, 3), "state needs 'v' shaped [batch, n]"
        fired = False
        for _ in range(200):
            s, st = nrn.step(st, torch.full((2, 3), drive), heaviside)
            _check_spikes(s, (2, 3))
            fired |= bool(s.any())
        assert fired == want, f"with input {drive} it {'fired' if fired else 'did not fire'}"


def synapse(cls):
    syn = cls(3, DT, **cls.defaults)
    st = syn.init_state(2)
    I, st = syn.step(st, torch.ones(2, 3))
    assert I.shape == (2, 3) and bool(torch.isfinite(I).all()), "current must be finite, shaped [batch, n]"
    total = I.clone()
    for _ in range(500):
        I, st = syn.step(st, torch.zeros(2, 3))
        total += I
    assert bool((total > 0).all()), "a positive input must give a positive current"


def topology(cls):
    m = cls(**cls.defaults).mask(4, 5, torch.Generator().manual_seed(0))
    assert m.dtype == torch.bool and m.shape == (4, 5), "mask must be bool [n_pre, n_post]"


def architecture(cls):
    arch = cls(**cls.defaults)
    net = arch.build(lambda n: _StubNeuron(n), lambda n: _StubSynapse(), _StubTopology(), DT, torch.Generator().manual_seed(0))
    sizes = net.sizes
    out = net.run(torch.ones(5, 2, sizes[0]))
    assert len(out["spikes"]) == len(sizes) - 1, "one spike record per non-input layer"
    for s, n in zip(out["spikes"], sizes[1:]):
        assert s.shape == (5, 2, n), f"layer record shaped {tuple(s.shape)}"


def rule(cls):
    r = cls(**cls.defaults)
    assert callable(getattr(r, "fit", None)), "rule needs fit(net, task, gen)"


def task(cls):
    t = cls(DT, **cls.defaults)
    x, y = t.test()
    assert x.ndim == 3 and x.shape[2] == t.n_in, "test input must be [T, batch, n_in]"
    assert (y is None) == (t.n_classes is None), "labels exactly when n_classes is set"


contracts.update(neuron=neuron, synapse=synapse, topology=topology, architecture=architecture, rule=rule, task=task)
