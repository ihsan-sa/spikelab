# spikelab

A small workbench for spiking neural networks: swap the neuron, synapse, topology, architecture and learning rule (STDP or surrogate gradients) from one TOML config or a local web page. The guide is [docs/GUIDE.md](docs/GUIDE.md) ([PDF](docs/spikelab-guide.pdf)).

```
python3 -m venv .venv && .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch && .venv/bin/pip install -e '.[dev]'
.venv/bin/spikelab run configs/surrogate.toml
.venv/bin/spikelab serve
```

`run` writes figures to the config's `out` folder. `serve` opens http://127.0.0.1:8765/ (loopback only).

Public mode (behind Cloudflare Access; see the guide, "Putting it online"). As a systemd user unit:
`cp public.example.toml public.toml` and fill it in; `cp deploy/spikelab.service ~/.config/systemd/user/`;
`systemctl --user daemon-reload && systemctl --user enable --now spikelab`. It serves 127.0.0.1:8765 only; stop any local `spikelab serve` on that port first.

Tests: `.venv/bin/pytest` (about 75 s). Guide figures: `.venv/bin/python scripts/make_figures.py`, then `.venv/bin/python scripts/build_doc.py` for the web screenshot and the PDF.
