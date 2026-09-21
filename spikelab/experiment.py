"""Run one config: build, train, record a sample, write the figures and metrics."""
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
    dt = cfg["dt"]
    plots.raster(x[:, :1], rec["spikes"], dt, out / "raster.png")
    plots.membrane(rec["v"][0], dt, out / "membrane.png", v_th=cfg["neuron"].get("v_th", 1.0))
    plots.weights(net, out / "weights.png")
    plots.curves(hist, out / "learning.png")
    final = {k: v[-1] for k, v in hist.items() if k != "x_label" and v}
    result = {"seconds": round(time.time() - t0, 1), "final": final, "history": hist,
              "figures": ["raster.png", "membrane.png", "weights.png", "learning.png"]}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    (out / "config.toml").write_text(config.dumps(cfg))
    return result
