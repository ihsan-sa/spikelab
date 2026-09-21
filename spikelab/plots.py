"""The four standard figures of a run."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

INK, BLUE, ORANGE = "#1f2937", "#2563eb", "#ea580c"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def raster(x, spikes, dt, path):
    """Input and every layer for the first sample. x [T,B,n], spikes list of [T,B,n]."""
    rows = [("input", x)] + [(f"layer {i + 1}", s) for i, s in enumerate(spikes)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7, 1.2 + 1.3 * len(rows)), sharex=True, squeeze=False)
    for ax, (label, s) in zip(axes[:, 0], rows):
        t, n = torch.nonzero(s[:, 0], as_tuple=True)
        ax.scatter(t.numpy() * dt * 1e3, n.numpy(), s=3, c=INK, marker="|")
        ax.set_ylabel(label)
        ax.set_ylim(-0.5, s.shape[2] - 0.5)
    axes[-1, 0].set_xlabel("time (ms)")
    axes[0, 0].set_title("Spike raster (first sample)")
    _save(fig, path)


def membrane(v, dt, path, v_th=1.0, k=3):
    """Membrane of the first k neurons of the first non-input layer."""
    fig, ax = plt.subplots(figsize=(7, 3))
    v = v.detach().numpy()
    t = torch.arange(v.shape[0]).numpy() * dt * 1e3
    for j in range(min(k, v.shape[2])):
        ax.plot(t, v[:, 0, j] + 1.5 * j, lw=1)
        ax.axhline(v_th + 1.5 * j, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("v (offset per neuron)")
    ax.set_title("Membrane potential, layer 1")
    _save(fig, path)


def weights(net, path):
    L = net.layers[0]
    W = L.W.detach()[L.mask]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3))
    im = a.imshow((L.W.detach() * L.mask).T.numpy(), aspect="auto", cmap="viridis")
    a.set_xlabel("input")
    a.set_ylabel("neuron")
    a.set_title("Layer 1 weights")
    fig.colorbar(im, ax=a)
    b.hist(W.numpy(), bins=40, color=BLUE)
    b.set_xlabel("weight")
    b.set_title("Weight histogram")
    _save(fig, path)


def curves(hist, path):
    keys = [k for k in hist if k != "x_label"]
    fig, ax = plt.subplots(figsize=(6, 3))
    for k, c in zip(keys, (BLUE, ORANGE, INK, "green")):
        ax.plot(range(1, len(hist[k]) + 1), hist[k], label=k, color=c, marker="o", ms=3)
    ax.set_xlabel(hist.get("x_label", "step"))
    ax.set_title("Learning")
    ax.legend()
    _save(fig, path)
