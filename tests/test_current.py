"""Task `current`: a current injected straight into the input neurons, through the same path as the f-I check."""
import pytest
import torch

from spikelab import checks, config
from spikelab.registry import ComponentError


def cfg(neuron="lif", rule="none", **task):
    return {"seed": 0, "dt": 1e-4,
            "task": {"name": "current", "n_in": 3, "steps": 5000, "batches": 1, **task},
            "neuron": {"name": neuron}, "synapse": {"name": "delta"}, "topology": {"name": "dense"},
            "architecture": {"name": "layered", "sizes": [3, 2]}, "rule": {"name": rule}}


def input_spikes(c):
    _, net, _, task, _ = config.build(c)
    x, _ = task.test()
    with torch.no_grad():
        return net.run(x)["input_spikes"][:, 0]  # [T, n_in]


def test_constant_current_gives_the_f_i_rate():
    for amp in (1.2, 2.0, 4.0):
        for tau, t_ref in ((0.02, 0.0), (0.01, 0.002)):  # the chosen neuron's own parameters
            c = cfg(amplitude=amp)
            c["neuron"] |= {"tau_mem": tau, "t_ref": t_ref}
            rate = checks.isi_rate(input_spikes(c), 1e-4)
            assert list(rate) == pytest.approx([float(checks.lif_rate_theory(amp, tau, t_ref))] * 3, rel=0.01)


def test_step_gives_no_spikes_before_onset():
    s = input_spikes(cfg(waveform="step", amplitude=3.0, onset=0.2, offset=0.4))
    on, off = 2000, 4000
    assert s[:on].sum() == 0 and s[on:off].sum() > 0
    assert s[off + 1:].sum() == 0  # the membrane can still be over threshold for one step after offset


def test_spread_makes_neurons_differ():
    rate = checks.isi_rate(input_spikes(cfg(amplitude=2.0, spread=0.3)), 1e-4)
    assert rate[0] < rate[1] < rate[2]


def test_adlif_adapts_under_constant_current():
    s = input_spikes(cfg("adlif", amplitude=2.0))
    first, last = s[:1000].sum(), s[-1000:].sum()
    assert first > 1.3 * last > 0


def test_ramp_and_noisy_run_with_stdp():
    for w in ("ramp", "noisy"):
        c = cfg(rule="stdp", waveform=w, steps=500)
        c["dt"] = 1e-3
        _, net, rule, task, gen = config.build(c)
        assert "input_rate_hz" in rule.fit(net, task, gen)


def test_surrogate_is_refused_by_name():
    with pytest.raises(ComponentError, match="rule 'surrogate' needs a labelled task; task 'current' has no labels"):
        config.build(cfg(rule="surrogate"))


def test_unknown_waveform_is_named():
    with pytest.raises(ValueError, match="waveform must be one of constant, step, ramp, noisy"):
        config.build(cfg(waveform="sine"))
