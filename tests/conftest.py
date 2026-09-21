import pytest

from spikelab import registry

registry.load_builtins()


@pytest.fixture
def tiny_patterns():
    """A small, fast surrogate config as a dict."""
    return {
        "seed": 0, "dt": 0.001,
        "task": {"name": "patterns", "n_in": 10, "n_classes": 2, "steps": 20, "train": 32, "test": 16, "batch": 16},
        "neuron": {"name": "lif"}, "synapse": {"name": "delta"}, "topology": {"name": "dense"},
        "architecture": {"name": "layered", "sizes": [10, 8, 2], "weight_scale": 30.0},
        "rule": {"name": "surrogate", "epochs": 1},
    }
