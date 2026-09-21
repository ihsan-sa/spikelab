import http.client
import json
import socket
import subprocess
import time
import threading

import pytest

from spikelab import config, limits, web


def serving(tmp_path, tiny_patterns, **kw):
    """A server on a free port, already serving. kw goes to web.make_server."""
    cfg = config.resolve(tiny_patterns)
    cfg["out"] = str(tmp_path / "run")
    p = tmp_path / "c.toml"
    p.write_text(config.dumps(cfg))
    srv = web.make_server(str(p), 0, **kw)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def server(tmp_path, tiny_patterns):
    srv = serving(tmp_path, tiny_patterns)
    yield srv
    srv.shutdown()


@pytest.fixture
def stingy(tmp_path, tiny_patterns):
    """The same server, but dropping a stalled connection after half a second and holding only two."""
    srv = serving(tmp_path, tiny_patterns, conn_timeout=0.5, max_connections=2)
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


def raw(srv, method, path="/", body=None, headers=None):
    """Status, headers and body, for the checks that read a header."""
    port = srv.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=90)
    c.request(method, path, body=body, headers={"Host": f"127.0.0.1:{port}", **(headers or {})})
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read()


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
    cfg["task"] = {"name": "correlated", "batches": 10.0}
    code, body = req(server, "POST", "/api/run", json.dumps(cfg))
    assert code == 400 and "task.batches must be a whole number" in json.loads(body)["error"]


@pytest.mark.parametrize("bad", [12.0, "12", True, float("nan"), float("inf"), -3, 0])
@pytest.mark.parametrize("where, change", [
    ("architecture.sizes[1]", lambda c, v: c["architecture"].update(sizes=[10, v, 2])),
    ("task.steps", lambda c, v: c["task"].update(steps=v)),
    ("task.train", lambda c, v: c["task"].update(train=v)),
    ("task.test", lambda c, v: c["task"].update(test=v)),
    ("task.batch", lambda c, v: c["task"].update(batch=v)),
    ("rule.epochs", lambda c, v: c["rule"].update(epochs=v)),
])
def test_a_capped_field_that_is_not_a_whole_number_is_refused_by_name(server, tiny_patterns, where, change, bad):
    change(tiny_patterns, bad)
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    assert code == 400 and where in json.loads(body)["error"], (where, bad)
    assert server.app.n == 0  # nothing ran


def test_floats_cannot_walk_past_the_activity_cap(server, tiny_patterns):
    """steps=1000.0, batch=128.0 used to pass every cap at 28x the activity limit."""
    tiny_patterns["task"].update(steps=1000.0, batch=128.0)
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    err = json.loads(body)["error"]
    assert code == 400 and "task.steps must be a whole number" in err and "task.batch must be" in err
    assert server.app.n == 0


def test_a_bad_dt_is_refused(server, tiny_patterns):
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns | {"dt": float("nan")}))
    assert code == 400 and "dt must be a positive number" in json.loads(body)["error"]


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
    err = json.loads(body)["error"]
    assert code == 400 and err == "the run failed"  # what died, and where, is the log's business
    assert "/" not in err and "Error" not in err and "Traceback" not in body.decode()
    assert req(server, "GET", "/api/state")[0] == 200


def test_the_childs_own_complaint_about_the_config_still_comes_back(server, tiny_patterns):
    """Only the run's own words about a field the caller sent survive the generic message above."""
    tiny_patterns["architecture"]["sizes"] = [9, 8, 2]  # the task has 10 inputs
    code, body = req(server, "POST", "/api/run", json.dumps(tiny_patterns))
    err = json.loads(body)["error"]
    assert code == 400 and "sizes[0]=9" in err and "10 inputs" in err


@pytest.mark.parametrize("length, want", [("-1", 400), ("abc", 400), ("", 400), ("9999999999", 413),
                                          (str(web.MAX_BODY + 1), 413)])
def test_a_bad_content_length_is_refused_and_the_server_carries_on(server, length, want):
    """-1 read to EOF (any body, any size) and letters raised out of do_POST with no answer at all."""
    code, body = req(server, "POST", "/api/run", None, headers={"Content-Length": length,
                                                               "Content-Type": "application/json"})
    assert code == want, (length, body)
    assert server.app.n == 0
    assert req(server, "GET", "/api/state")[0] == 200


def test_a_stalled_connection_is_dropped(stingy):
    s = socket.create_connection(("127.0.0.1", stingy.server_address[1]), timeout=30)
    try:
        s.sendall(b"GET / HTTP")  # half a request line, then nothing ever again
        t0 = time.time()
        assert s.recv(100) == b""  # the server gave up on it and closed
        assert 0.3 < time.time() - t0 < 20
    finally:
        s.close()
    assert req(stingy, "GET", "/api/state")[0] == 200


def test_only_so_many_connections_are_held_at_once(tmp_path, tiny_patterns):
    srv = serving(tmp_path, tiny_patterns, conn_timeout=20, max_connections=2)
    held = []
    try:
        for _ in range(2):
            s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=30)
            s.sendall(b"GET /api/state HTTP/1.0\r\n")  # started, never finished: each holds a thread
            held.append(s)
        time.sleep(0.5)
        extra = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=30)
        held.append(extra)
        assert extra.recv(100) == b""  # over the cap: closed at the door, no thread of its own
        for s in held:
            s.close()
        held = []
        for _ in range(100):  # the two threads end, the room comes back
            try:
                if req(srv, "GET", "/api/state")[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        else:
            pytest.fail("the server never took a connection again")
    finally:
        for s in held:
            s.close()
        srv.shutdown()


@pytest.mark.parametrize("method", ["HEAD", "OPTIONS", "PUT", "DELETE", "PATCH", "TRACE", "WHAT"])
def test_every_other_method_is_405_and_names_no_version(server, method):
    code, headers, body = raw(server, method, "/")
    assert code == 405
    assert headers["Server"] == "spikelab" and "Python" not in headers["Server"]
    assert b"Unsupported" not in body and b"Traceback" not in body


def test_a_foreign_host_is_still_403_on_any_method(server):
    assert raw(server, "PUT", "/", headers={"Host": "evil.example"})[0] == 403


def test_an_oversized_request_line_names_no_version_either(server):
    """The base class answers this one itself (414), which used to carry BaseHTTP and the Python version."""
    line = b"GET /" + b"x" * (65537 - 7) + b"\r\n"  # 65537 bytes: all of it read, none left over
    s = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=30)
    try:
        s.sendall(line)
        head = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            head += chunk
    finally:
        s.close()
    assert b" 414 " in head and b"Server: spikelab\r\n" in head
    assert b"Python/" not in head and b"BaseHTTP" not in head
