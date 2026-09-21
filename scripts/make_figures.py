"""Draw every figure in docs/GUIDE.md: the architecture diagram and the three correctness checks.

    .venv/bin/python scripts/make_figures.py      (about a minute)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from spikelab import checks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"
INK, BLUE, ORANGE, GREY = "#1f2937", "#2563eb", "#ea580c", "#9ca3af"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=130)
    plt.close(fig)
    print("wrote", IMG / name)


def box(ax, x, y, w, h, text, color="#eff6ff", edge=BLUE, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=edge, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, a, b, color=INK, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=1.2,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def architecture():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    # config and registries
    box(ax, 0.2, 3.9, 2.0, 0.8, "config.toml\n(or the web page)", "#fff7ed", ORANGE, 10)
    kinds = ["neuron", "synapse", "topology", "architecture", "rule", "task"]
    for i, k in enumerate(kinds):
        box(ax, 0.2, 3.2 - i * 0.52, 2.0, 0.42, f"{k} registry", size=8.5)
    arrow(ax, (1.2, 3.9), (1.2, 3.64))
    ax.text(1.2, 0.05, "each part checked on load;\na broken one is named, the rest work", ha="center", fontsize=7.5, color=GREY)
    # engine
    ax.add_patch(FancyBboxPatch((2.9, 0.9), 4.6, 3.9, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=INK, lw=1.0, ls="--"))
    ax.text(5.2, 4.6, "engine: one step of dt, every tensor [batch, n]", ha="center", fontsize=9.5, color=INK)
    steps = [("input spikes", 3.8), ("× weights · topology mask", 3.1), ("synapse → current", 2.4), ("neuron → spikes", 1.7)]
    for text, y in steps:
        box(ax, 3.4, y, 2.8, 0.5, text, size=9)
    for (_, y1), (_, y2) in zip(steps[:-1], steps[1:]):
        arrow(ax, (4.8, y1), (4.8, y2 + 0.5))
    arrow(ax, (6.2, 1.95), (6.2, 3.35), BLUE, rad=0.6)
    ax.text(7.05, 2.65, "next layer\n(and own layer\nif recurrent)", fontsize=7.5, color=BLUE, ha="center")
    ax.text(5.2, 1.1, "repeat for each layer, then the next time step", ha="center", fontsize=8, color=GREY)
    arrow(ax, (2.2, 2.4), (3.4, 2.65), GREY)
    # rules
    box(ax, 8.0, 3.3, 1.9, 0.9, "STDP\nupdates weights\nafter each step", "#f0fdf4", "green", 8.5)
    box(ax, 8.0, 1.9, 1.9, 0.9, "surrogate gradient\nbackprop through\nall steps", "#f0fdf4", "green", 8.5)
    arrow(ax, (8.0, 3.75), (6.2, 3.35), "green", ls="--")
    arrow(ax, (8.0, 2.35), (6.2, 1.95), "green", ls="--")
    box(ax, 8.0, 0.5, 1.9, 0.8, "figures + metrics\nin the out folder", "#fff7ed", ORANGE, 8.5)
    arrow(ax, (7.5, 1.2), (8.0, 0.9))
    save(fig, "architecture.png")


def fi_curve():
    I = np.linspace(0.5, 5.0, 19)
    I_fine = np.linspace(0.5, 5.0, 400)
    sim = checks.lif_rate_sim(I, dt=1e-4)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(I_fine, checks.lif_rate_theory(I_fine), color=GREY, lw=2, label="theory")
    ax.plot(I, sim, "o", color=BLUE, ms=5, label="engine")
    ax.set_xlabel("input current R·I (threshold = 1)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title("LIF f-I curve, τ = 20 ms, t_ref = 2 ms")
    ax.legend()
    save(fig, "fi_curve.png")


def stdp_window():
    p = dict(A_plus=0.01, A_minus=0.0135, tau_plus=0.02, tau_minus=0.02)
    d = np.arange(-60, 61, 4)
    d_fine = np.linspace(-60, 60, 600)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    th = checks.stdp_theory(d_fine, **p)
    ax.plot(d_fine[d_fine < 0], th[d_fine < 0], color=GREY, lw=2, label="theory")
    ax.plot(d_fine[d_fine >= 0], th[d_fine >= 0], color=GREY, lw=2)
    ax.plot(d, checks.stdp_window(d, **p), "o", color=BLUE, ms=5, label="engine")
    ax.axhline(0, color=INK, lw=0.5)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_xlabel("t_post − t_pre (ms)")
    ax.set_ylabel("weight change")
    ax.set_title("STDP window from one spike pair")
    ax.legend()
    save(fig, "stdp_window.png")


def stdp_task():
    net, task, hist = checks.run_config(ROOT / "configs" / "stdp.toml")
    w = net.layers[0].W.detach()[:, 0].numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(1, len(hist["w_correlated"]) + 1)
    a.plot(x, hist["w_correlated"], color=ORANGE, label="correlated half")
    a.plot(x, hist["w_uncorrelated"], color=BLUE, label="independent half")
    a.set_xlabel("batch (8 × 1 s)")
    a.set_ylabel("mean weight")
    a.set_ylim(0, 1.05)
    a.legend()
    a.set_title("Weights during training")
    bins = np.linspace(0, 1, 26)
    b.hist(w[: task.half], bins=bins, color=ORANGE, alpha=0.8, label="correlated")
    b.hist(w[task.half:], bins=bins, color=BLUE, alpha=0.8, label="independent")
    b.set_xlabel("final weight (w_max = 1)")
    b.set_ylabel("inputs")
    b.legend()
    b.set_title(f"Bimodal: {checks.bimodality(w, 1.0):.0%} within 10% of a bound")
    save(fig, "stdp_task.png")


def surrogate_task():
    net, task, hist = checks.run_config(ROOT / "configs" / "surrogate.toml")
    x = np.arange(1, len(hist["loss"]) + 1)
    fig, a = plt.subplots(figsize=(5.5, 3.4))
    a.plot(x, hist["loss"], color=INK, marker="o", ms=3, label="training loss")
    a.set_xlabel("epoch")
    a.set_ylabel("cross-entropy")
    b = a.twinx()
    b.plot(x, hist["test_accuracy"], color=ORANGE, marker="o", ms=3, label="test accuracy")
    b.axhline(0.25, color=GREY, ls=":", lw=1)
    b.text(x[-1], 0.27, "chance", ha="right", fontsize=8, color=GREY)
    b.set_ylim(0, 1.05)
    b.set_ylabel("test accuracy")
    fig.legend(loc="center right", bbox_to_anchor=(0.85, 0.5), fontsize=8)
    a.set_title(f"Surrogate gradient: {hist['test_accuracy'][-1]:.0%} on held-out patterns")
    save(fig, "surrogate_task.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    architecture()
    fi_curve()
    stdp_window()
    stdp_task()
    surrogate_task()
