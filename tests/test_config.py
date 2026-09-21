import json
import tomllib

import pytest

from spikelab import cli, config
from spikelab.registry import ComponentError


def test_errors_name_the_component_and_are_reported_together(tiny_patterns):
    tiny_patterns["synapse"] = {"name": "delta", "tau_syn": 0.01}
    tiny_patterns["topology"] = {"name": "nope"}
    with pytest.raises(ComponentError) as e:
        config.resolve(tiny_patterns)
    assert "synapse 'delta' has no parameter" in str(e.value)
    assert "unknown topology 'nope'" in str(e.value)


def test_sizes_must_match_the_task(tiny_patterns):
    tiny_patterns["architecture"]["sizes"] = [11, 2]
    with pytest.raises(ComponentError, match="sizes"):
        config.build(tiny_patterns)


def test_dumps_round_trips(tiny_patterns):
    cfg = config.resolve(tiny_patterns)
    assert tomllib.loads(config.dumps(cfg)) == cfg


def test_cli_run_writes_the_four_figures(tmp_path, tiny_patterns, capsys):
    path = tmp_path / "c.toml"
    path.write_text(config.dumps(config.resolve(tiny_patterns)))
    assert cli.main(["run", str(path), "--out", str(tmp_path / "out")]) == 0
    for f in ("raster", "membrane", "weights", "learning"):
        assert (tmp_path / "out" / f"{f}.png").stat().st_size > 1000
    assert "test_accuracy" in json.loads(capsys.readouterr().out)["final"]


def test_shipped_configs_resolve():
    for p in ("configs/surrogate.toml", "configs/stdp.toml"):
        config.resolve(config.load(p))
