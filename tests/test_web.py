import http.client
import json
import subprocess
import time
import threading

import pytest

from spikelab import config, limits, web


@pytest.fixture
def server(tmp_path, tiny_patterns):
    cfg = config.resolve(tiny_patterns)
    cfg["out"] = str(tmp_path / "run")
    p = tmp_path / "c.toml"
    p.write_text(config.dumps(cfg))
    srv = web.make_server(str(p), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def req(srv, method, path, body=None, host=None, origin=None, ctype="application/json", headers=None):
    port = srv.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=90)
    headers = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    if body is not None and ctype is not None:
        headers["Content-Type"] = ctype
    if origin is not None:
        headers["Origin"] = origin
    c.request(method, path, body=body, headers=headers)
    r = c.getresponse()
    return r.status, r.read()


def test_binds_loopback_and_refuses_foreign_host(server):
    assert server.server_address[0] == "127.0.0.1"
    assert req(server, "GET", "/", host="evil.example")[0] == 403


def test_local_mode_refuses_the_public_hostname(server):
    assert req(server, "GET", "/", host="spike.ihsan.cc")[0] == 403
    assert req(server, "GET", "/guide.pdf", host="spike.ihsan.cc")[0] == 403


def test_serves_the_guide(server):
    code, pdf = req(server, "GET", "/guide.pdf")
    assert code == 200 and pdf[:5] == b"%PDF-"
    assert b'href="/guide.pdf"' in req(server, "GET", "/")[1]


def test_refuses_another_sites_origin(server, tiny_patterns):
    body = json.dumps(tiny_patterns)
    assert req(server, "POST", "/api/run", body, origin="http://evil.example")[0] == 403
    assert req(server, "GET", "/", origin="http://evil.example")[0] == 403
    port = server.server_address[1]
    for good in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
        assert req(server, "GET", "/api/state", origin=good)[0] == 200


def test_refuses_a_post_that_is_not_json(server, tiny_patterns):
    port = server.server_address[1]
    body, origin = json.dumps(tiny_patterns), f"http://127.0.0.1:{port}"
    assert req(server, "POST", "/api/run", body, origin=origin, ctype="text/plain")[0] == 415
    assert req(server, "POST", "/api/run", body, origin=origin, ctype=None)[0] == 415
    code, _ = req(server, "POST", "/api/run", body, origin=origin,
                  ctype="application/json; charset=utf-8")
    assert code == 200


def test_page_lists_registries_and_runs(server, tiny_patterns):
    code, body = req(server, "GET", "/api/state")
    state = json.loads(body)
    assert code == 200 and {"lif", "adlif"} <= set(state["registry"]["neuron"]["ok"])
    tiny_patterns["neuron"] = {"name": "adlif"}
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    res = json.loads(body)
    assert code == 200, res
    assert 'name = "adlif"' in res["toml"]
    code, png = req(server, "GET", res["figures"][0])
    assert code == 200 and png[:4] == b"\x89PNG"


def test_bad_config_is_a_400_naming_it(server, tiny_patterns):
    tiny_patterns["rule"] = {"name": "nope"}
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    assert code == 400 and "unknown rule 'nope'" in json.loads(body)["error"]


@pytest.mark.parametrize("cap, change", [
    ("layer_size", lambda c: c["architecture"].update(sizes=[10, 300, 2])),
    ("weights", lambda c: c["architecture"].update(sizes=[10, 200, 200, 2])),
    ("steps", lambda c: c["task"].update(steps=5000)),
    ("train", lambda c: c["task"].update(train=5000)),
    ("test", lambda c: c["task"].update(test=5000)),
    ("batch", lambda c: c["task"].update(batch=1000)),
    ("epochs", lambda c: c["rule"].update(epochs=500)),
    ("activity", lambda c: c["task"].update(steps=1000, batch=128)),
])
def test_each_cap_refuses_by_name_before_running(server, tiny_patterns, cap, change):
    change(tiny_patterns)
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    assert code == 400 and f"over the {cap} limit" in json.loads(body)["error"]
    assert server.app.n == 0  # nothing ran


def test_batches_cap(server):
    cfg = config.load("configs/stdp.toml") | {"task": {"name": "correlated", "batches": 500}}
    code, body = req(server, "POST", "/api/run", json.dumps(cfg))
    assert code == 400 and "over the batches limit" in json.loads(body)["error"]


def test_the_example_configs_fit_the_caps():
    for f in ("configs/surrogate.toml", "configs/stdp.toml"):
        limits.check(config.resolve(config.load(f)))


def test_one_run_at_a_time(server, tiny_patterns):
    server.app.lock.acquire()
    try:
        code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    finally:
        server.app.lock.release()
    assert code == 409 and "busy" in json.loads(body)["error"]


def test_time_limit_kills_the_run_and_the_server_carries_on(server, tiny_patterns):
    server.app.seconds = 2
    t0 = time.time()
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns | {"rule": {"name": "surrogate", "epochs": 30}}))
    assert code == 400 and "2 s time limit" in json.loads(body)["error"]
    assert time.time() - t0 < 10
    out = server.app.root / "1"
    time.sleep(1)
    assert not (out / "metrics.json").exists()
    assert subprocess.run(["pgrep", "-f", f"spikelab.worker {out}"]).returncode == 1  # no worker left
    assert req(server, "GET", "/api/state")[0] == 200
    server.app.seconds = 60
    assert req(server, "POST", "/api/run", json.dumps(tiny_patterns))[0] == 200


def test_a_crash_in_the_run_is_a_400_not_a_dead_server(server, tiny_patterns):
    tiny_patterns["neuron"] = {"name": "lif", "tau_mem": 0.0}
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    assert code == 400 and json.loads(body)["error"]
    assert req(server, "GET", "/api/state")[0] == 200
