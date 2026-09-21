"""The clock-driven engine. Fixed dt, every tensor shaped [batch, n].

One step, per layer: input spikes -> weights (masked by the topology) -> synapse -> neuron -> spikes,
which are the next layer's input. Recurrent layers also see their own spikes from the previous step.
The engine knows nothing about which neuron, synapse, topology or rule it runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .spike import heaviside


@dataclass
class Layer:
    W: torch.nn.Parameter  # [n_pre, n_post]
    mask: torch.Tensor  # bool, same shape
    neuron: object
    synapse: object
    V: torch.nn.Parameter | None = None  # recurrent [n_post, n_post]
    vmask: torch.Tensor | None = None


class Network(torch.nn.Module):
    def __init__(self, layers: list[Layer], dt: float):
        super().__init__()
        self.layers, self.dt = layers, dt
        self.params = torch.nn.ParameterList([p for L in layers for p in (L.W, L.V) if p is not None])
        self.spike_fn = heaviside  # a gradient rule swaps in a surrogate

    @property
    def sizes(self) -> list[int]:
        return [self.layers[0].W.shape[0]] + [L.W.shape[1] for L in self.layers]

    def run(self, x: torch.Tensor, record: bool = True, on_step=None) -> dict:
        """x: input spikes [T, batch, n_in]. Returns spikes and membrane per layer, each [T, batch, n].

        on_step(layer_index, pre_spikes, post_spikes) is called after each layer's step (used by STDP).
        """
        T, B, _ = x.shape
        nst = [L.neuron.init_state(B) for L in self.layers]
        sst = [L.synapse.init_state(B) for L in self.layers]
        prev = [torch.zeros(B, L.W.shape[1]) for L in self.layers]
        spikes = [[] for _ in self.layers]
        volts = [[] for _ in self.layers]
        for t in range(T):
            inp = x[t]
            for i, L in enumerate(self.layers):
                cur = inp @ (L.W * L.mask)
                if L.V is not None:
                    cur = cur + prev[i] @ (L.V * L.vmask)
                I, sst[i] = L.synapse.step(sst[i], cur)
                s, nst[i] = L.neuron.step(nst[i], I, self.spike_fn)
                if on_step is not None:
                    on_step(i, inp, s)
                if record:
                    spikes[i].append(s)
                    volts[i].append(nst[i]["v"])
                prev[i] = s
                inp = s
        out = {"last": prev}
        if record:
            out["spikes"] = [torch.stack(s) for s in spikes]
            out["v"] = [torch.stack(v) for v in volts]
        return out
