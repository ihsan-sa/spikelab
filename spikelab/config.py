"""One TOML file describes a whole experiment. Example: configs/surrogate.toml.

Top level: `seed`, `dt` (seconds), `out` (output folder), optional `plugins` (files that register
components). One table per swappable part — [neuron] [synapse] [topology] [architecture] [rule] —
plus [task]. Each table has `name` (a registered component) and that component's parameters.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import torch

from . import registry
from .registry import ComponentError

SECTIONS = ("neuron", "synapse", "topology", "architecture", "rule", "task")
TOP = {"seed": 0, "dt": 0.001, "out": "out/run", "plugins": []}


def load(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    base = Path(path).parent
    cfg["plugins"] = [str((base / p).resolve()) for p in cfg.get("plugins", [])]
    return cfg


def resolve(cfg: dict) -> dict:
    """Fill defaults and check every section. All problems are reported together, each by name."""
    registry.load_builtins()
    for p in cfg.get("plugins", []):
        registry.load_plugin(p)
    unknown = set(cfg) - set(SECTIONS) - set(TOP)
    errors = [f"unknown top-level key(s) {sorted(unknown)}"] if unknown else []
    out = {k: cfg.get(k, v) for k, v in TOP.items()}
    for kind in SECTIONS:
        sec = dict(cfg.get(kind, {}))
        name = sec.pop("name", None)
        if name is None:
            errors.append(f"[{kind}] needs a name; one of {registry.names(kind)}")
            continue
        try:
            out[kind] = {"name": name, **registry.params_for(kind, name, sec)}
        except ComponentError as e:
            errors.append(str(e))
    if errors:
        raise ComponentError("; ".join(errors))
    return out


def _params(sec: dict) -> dict:
    return {k: v for k, v in sec.items() if k != "name"}


def build(cfg: dict):
    """Resolved config -> (net, rule, task, gen). Nothing here knows which components it builds."""
    cfg = resolve(cfg)
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    dt = cfg["dt"]
    get = registry.get
    neuron_cls, np_ = get("neuron", cfg["neuron"]["name"]), _params(cfg["neuron"])
    syn_cls, sp = get("synapse", cfg["synapse"]["name"]), _params(cfg["synapse"])
    topo = get("topology", cfg["topology"]["name"])(**_params(cfg["topology"]))
    task = get("task", cfg["task"]["name"])(dt, **_params(cfg["task"]))
    arch = get("architecture", cfg["architecture"]["name"])(**_params(cfg["architecture"]))
    if arch.p.get("sizes", [task.n_in])[0] != task.n_in:
        raise ComponentError(f"architecture sizes[0]={arch.p['sizes'][0]} but task {task.name!r} has {task.n_in} inputs")
    net = arch.build(lambda n: neuron_cls(n, dt, **np_), lambda n: syn_cls(n, dt, **sp), topo, dt, gen)
    if getattr(task, "drive", "spikes") == "current":
        net.input_neuron = neuron_cls(task.n_in, dt, **np_)
    rule_cls = get("rule", cfg["rule"]["name"])
    if getattr(rule_cls, "needs_labels", False) and task.n_classes is None:
        raise ComponentError(f"rule {rule_cls.name!r} needs a labelled task; task {task.name!r} has no labels"
                             " (use rule 'stdp' or 'none')")
    rule = rule_cls(**_params(cfg["rule"]))
    return cfg, net, rule, task, gen


def dumps(cfg: dict) -> str:
    """Write a resolved config back as TOML (flat values and lists only, which is all a config holds)."""

    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return repr(v)

    lines = [f"{k} = {val(cfg[k])}" for k in TOP if k in cfg]
    for kind in SECTIONS:
        if kind in cfg:
            lines += ["", f"[{kind}]"] + [f"{k} = {val(v)}" for k, v in cfg[kind].items()]
    return "\n".join(lines) + "\n"
