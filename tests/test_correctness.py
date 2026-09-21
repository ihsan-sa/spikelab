"""The three known results. scripts/make_figures.py draws the same numbers for docs/GUIDE.md."""
import numpy as np
import pytest

from spikelab import checks

DT = 1e-4


def test_lif_fi_curve_matches_theory():
    I = np.linspace(0.5, 5.0, 19)
    sim, th = checks.lif_rate_sim(I, dt=DT), checks.lif_rate_theory(I)
    assert ((sim == 0) == (th == 0)).all(), "fires exactly above rheobase (RI > v_th)"
    on = th > 0
    # the clock rounds each interval up to a whole step, so periods agree to within one dt
    assert np.abs(1 / sim[on] - 1 / th[on]).max() <= 1.01 * DT


def test_stdp_window_matches_theory():
    p = dict(A_plus=0.01, A_minus=0.0135, tau_plus=0.02, tau_minus=0.02)
    d = np.arange(-60, 61, 5)
    assert checks.stdp_window(d, **p) == pytest.approx(checks.stdp_theory(d, **p), abs=1e-7)


def test_stdp_picks_out_the_correlated_group():
    net, task, hist = checks.run_config("configs/stdp.toml")
    w = net.layers[0].W.detach()[:, 0].numpy()
    assert w[: task.half].mean() > 0.8 and w[task.half :].mean() < 0.2
    assert checks.bimodality(w, 1.0) > 0.8


def test_surrogate_gradient_classifies_patterns():
    net, task, hist = checks.run_config("configs/surrogate.toml")
    assert hist["test_accuracy"][-1] > 0.9  # chance is 0.25
    assert hist["loss"][-1] < hist["loss"][0]
