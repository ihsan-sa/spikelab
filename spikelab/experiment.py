"""Run one config: build, train, record a sample, write the figures, metrics and learned weights.

weights.json: {"layers": [{"W": [n_pre][n_post], "V": [n][n] or null}]}, a connection the topology
left out is null. The web page draws it on the network diagram.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from . import config, plots


def run(cfg: dict, out: str | Path | None = None) -> dict:
    cfg, net, rule, task, gen = config.build(cfg)
    out = Path(out or cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hist = rule.fit(net, task, gen)
    x, _ = task.test()
    with torch.no_grad():
        rec = net.run(x[:, :1])
    dt, v_th = cfg["dt"], cfg["neuron"].get("v_th", 1.0)
    if "input_spikes" in rec:  # a current task: show the driven input neurons under their current
        cur = x[:, :1]
        plots.raster(rec["input_spikes"], rec["spikes"], dt, out / "raster.png", current=cur)
        plots.membrane(rec["input_v"], dt, out / "membrane.png", v_th=v_th, current=cur, title="input neurons")
    else:
        plots.raster(x[:, :1], rec["spikes"], dt, out / "raster.png")
        plots.membrane(rec["v"][0], dt, out / "membrane.png", v_th=v_th)
    plots.weights(net, out / "weights.png")
    plots.curves(hist, out / "learning.png")
    final = {k: v[-1] for k, v in hist.items() if k != "x_label" and v}
    result = {"seconds": round(time.time() - t0, 1), "final": final, "history": hist,
              "figures": ["raster.png", "membrane.png", "weights.png", "learning.png"]}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    (out / "config.toml").write_text(config.dumps(cfg))
    (out / "weights.json").write_text(json.dumps({"layers": [
        {"W": _masked(L.W, L.mask), "V": None if L.V is None else _masked(L.V, L.vmask)} for L in net.layers]}))
    return result


def _masked(W, mask) -> list:
    return [[round(w, 4) if m else None for w, m in zip(rw, rm)] for rw, rm in zip(W.detach().tolist(), mask.tolist())]
