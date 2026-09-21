"""One contract test per kind of component, plus a broken component that must not spread."""
import textwrap

import pytest
import torch

from spikelab import config, contracts, registry
from spikelab.registry import ComponentError
from spikelab.spike import heaviside


@pytest.mark.parametrize("kind", registry.KINDS)
def test_every_registered_component_meets_its_contract(kind):
    assert registry.names(kind), f"no {kind} registered"
    assert not registry.broken[kind], registry.broken[kind]
    for name in registry.names(kind):
        getattr(contracts, kind)(registry.get(kind, name))


def test_neuron_adaptation_lowers_the_rate():
    def count(name, steps=1000):
        n = registry.get("neuron", name)(1, 1e-3)
        st, total = n.init_state(1), 0.0
        for _ in range(steps):
            s, st = n.step(st, torch.full((1, 1), 2.0), heaviside)
            total += s.item()
        return total

    assert count("adlif") < 0.8 * count("lif")


def test_synapses_deliver_the_same_charge():
    totals = []
    for name in registry.names("synapse"):
        syn = registry.get("synapse", name)(1, 1e-3)
        st = syn.init_state(1)
        I, st = syn.step(st, torch.ones(1, 1))
        tot = I.item()
        for _ in range(1000):
            I, st = syn.step(st, torch.zeros(1, 1))
            tot += I.item()
        totals.append(tot)
    assert totals == pytest.approx([1.0] * len(totals), rel=1e-3)


def test_sparse_topology_density():
    m = registry.get("topology", "sparse")(p=0.3).mask(200, 200, torch.Generator().manual_seed(0))
    assert m.float().mean().item() == pytest.approx(0.3, abs=0.02)
    with pytest.raises(ValueError):
        registry.get("topology", "sparse")(p=1.5)


def test_recurrent_flag_adds_recurrent_weights_to_hidden_layers_only(tiny_patterns):
    for flag in (False, True):
        tiny_patterns["architecture"]["recurrent"] = flag
        _, net, *_ = config.build(tiny_patterns)
        assert (net.layers[0].V is not None) == flag
        assert net.layers[-1].V is None
        if flag:
            assert not net.layers[0].vmask.diagonal().any()


def test_broken_components_are_named_and_the_rest_still_work(tmp_path, tiny_patterns):
    bad = tmp_path / "bad_neuron.py"
    bad.write_text(textwrap.dedent('''
        from spikelab.registry import register
        @register("neuron", "exploding")
        class Exploding:
            defaults = {}
            def __init__(self, n, dt, **p): pass
            def init_state(self, batch): raise RuntimeError("boom")
    '''))
    syntax = tmp_path / "typo.py"
    syntax.write_text("def (:\n")
    assert registry.load_plugin(bad) is None
    assert "SyntaxError" in registry.load_plugin(syntax)
    assert "boom" in registry.broken["neuron"]["exploding"]
    assert str(syntax) in registry.broken["module"]
    # asking for it names it
    tiny_patterns["neuron"] = {"name": "exploding"}
    with pytest.raises(ComponentError, match="neuron 'exploding' is broken"):
        config.build(tiny_patterns)
    # everything else still builds and runs
    tiny_patterns["neuron"] = {"name": "lif"}
    _, net, rule, task, gen = config.build(tiny_patterns)
    assert "loss" in rule.fit(net, task, gen)
    assert set(registry.names("neuron")) >= {"lif", "adlif"}
    del registry.broken["neuron"]["exploding"], registry.broken["module"][str(syntax)]
