import hashlib
import json
import sys
import textwrap

import httpx
import pytest

from qtrix_engine.localai import hub
from qtrix_engine.localai.jobs import Job, JobManager, download
from qtrix_engine.localai.llamacpp import LlamaRuntime, asset_names, pick_release
from qtrix_engine.localai.models import META_SUFFIX, ModelStore


def test_group_gguf_merges_split_parts_and_detects_quant():
    groups = hub.group_gguf([
        ("Qwen3-8B-Q4_K_M.gguf", 5_000, "a"),
        ("big/Model-Q8_0-00002-of-00002.gguf", 300, "c"),
        ("big/Model-Q8_0-00001-of-00002.gguf", 700, "b"),
        ("mmproj-model-f16.gguf", 50, "d"),
        ("README.md", 1, None),
    ])
    by_name = {g.name: g for g in groups}
    split = by_name["big/Model-Q8_0-00001-of-00002.gguf"]
    assert split.files == ["big/Model-Q8_0-00001-of-00002.gguf", "big/Model-Q8_0-00002-of-00002.gguf"]
    assert split.size == 1000 and split.quant == "Q8_0"
    assert by_name["Qwen3-8B-Q4_K_M.gguf"].quant == "Q4_K_M"
    assert by_name["mmproj-model-f16.gguf"].kind == "mmproj"
    assert groups[-1].kind == "mmproj"  # projectors listed last


def test_release_selection_skips_non_build_tags_and_missing_assets():
    releases = [
        {"tag_name": "v0.6.0", "assets": [{"name": "nightly-tag.txt"}]},
        {"tag_name": "b200", "assets": [{"name": "llama-b200-bin-win-cpu-x64.zip"}]},
        {"tag_name": "b199", "assets": [{"name": "llama-b199-bin-win-vulkan-x64.zip"},
                                        {"name": "llama-b199-bin-win-cpu-x64.zip"}]},
    ]
    assert pick_release(releases, "cpu", "win-x64")[0] == "b200"
    assert pick_release(releases, "vulkan", "win-x64")[0] == "b199"
    with pytest.raises(LookupError):
        pick_release(releases, "cuda", "win-x64")
    assert asset_names("b1", "cuda", "win-x64") == ["llama-b1-bin-win-cuda-12.4-x64.zip",
                                                     "cudart-llama-bin-win-cuda-12.4-x64.zip"]
    assert asset_names("b1", "vulkan", "macos-arm64") == ["llama-b1-bin-macos-arm64.tar.gz"]
    assert asset_names("b1", "cpu", "linux-x64") == ["llama-b1-bin-ubuntu-x64.tar.gz"]


def _range_server(payload: bytes, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range")
        seen.append(rng)
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=payload[start:])
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_resumes_and_verifies_sha256(tmp_path):
    payload = bytes(range(256)) * 1000
    sha = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "m.gguf"
    (tmp_path / "m.gguf.part").write_bytes(payload[:100_000])  # interrupted earlier
    seen: list = []
    job = Job(id="j", kind="model", title="t", total=len(payload))

    download("https://hf.test/m.gguf", dest, job, sha256=sha, client=_range_server(payload, seen))

    assert dest.read_bytes() == payload
    assert seen == ["bytes=100000-"]
    assert job.done == len(payload)
    assert json.dumps(job.to_dict())  # API-serialisable (the cancel Event must not leak into it)
    assert job.to_dict()["progress"] == 1.0
    assert not (tmp_path / "m.gguf.part").exists()


def test_download_rejects_corrupt_file(tmp_path):
    job = Job(id="j", kind="model", title="t")
    with pytest.raises(ValueError, match="SHA-256"):
        download("https://hf.test/x.gguf", tmp_path / "x.gguf", job, sha256="0" * 64,
                 client=_range_server(b"not the model", []))
    assert not (tmp_path / "x.gguf").exists() and not (tmp_path / "x.gguf.part").exists()


def test_model_store_lists_resolves_and_deletes(tmp_path):
    store = ModelStore(tmp_path, JobManager())
    repo_dir = tmp_path / "Qwen__Qwen3-4B-GGUF"
    repo_dir.mkdir()
    (repo_dir / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x" * 10)
    (repo_dir / ("Qwen3-4B-Q4_K_M.gguf" + META_SUFFIX)).write_text(json.dumps(
        {"repo": "Qwen/Qwen3-4B-GGUF", "files": ["Qwen3-4B-Q4_K_M.gguf"], "quant": "Q4_K_M"}))
    (tmp_path / "manual").mkdir()
    (tmp_path / "manual" / "Mine-Q5_K_M-00001-of-00002.gguf").write_bytes(b"a" * 4)
    (tmp_path / "manual" / "Mine-Q5_K_M-00002-of-00002.gguf").write_bytes(b"b" * 6)

    models = {m["id"]: m for m in store.list()}
    assert models["Qwen__Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"]["repo"] == "Qwen/Qwen3-4B-GGUF"
    manual = models["manual/Mine-Q5_K_M-00001-of-00002.gguf"]
    assert manual["size"] == 10 and manual["name"] == "Mine-Q5_K_M"
    assert len(models) == 2  # second split part is not listed on its own

    with pytest.raises(FileNotFoundError):
        store.resolve("../outside.gguf")
    store.delete("Qwen__Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf")
    assert not repo_dir.exists()


FAKE_SERVER = textwrap.dedent("""
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port = int(sys.argv[sys.argv.index("--port") + 1])
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        def log_message(self, *a):
            pass
    print("fake llama-server", " ".join(sys.argv[1:]), flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
""")


def test_runtime_serves_switches_and_stops(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    rt = LlamaRuntime(tmp_path / "runtime", tmp_path / "logs", JobManager(), server_cmd=[sys.executable, str(script)])
    try:
        url = rt.ensure(a, ctx_size=2048, timeout=30)
        assert url.startswith("http://127.0.0.1:") and url.endswith("/v1")
        assert rt.is_serving(a, 2048) and rt.ensure(a, ctx_size=2048) == url  # reused, no restart
        first_pid = rt._proc.pid
        rt.ensure(b, ctx_size=2048, timeout=30)
        assert rt.is_serving(b, 2048) and rt._proc.pid != first_pid
        log = (tmp_path / "logs" / "llama-server.log").read_text(encoding="utf-8", errors="replace")
        assert "--jinja" in log and "b.gguf" in log
    finally:
        rt.stop()
    assert not rt.running()


@pytest.mark.skipif(sys.platform == "darwin", reason="no kernel-level parent-death hook on macOS")
def test_server_dies_when_engine_is_killed_hard(tmp_path):
    """A hard-killed engine (no atexit) must not leave llama-server running."""
    import subprocess
    import time

    from qtrix_engine.__main__ import pid_alive

    (tmp_path / "fake_server.py").write_text(FAKE_SERVER)
    (tmp_path / "m.gguf").write_bytes(b"m")
    engine_like = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        from qtrix_engine.localai.jobs import JobManager
        from qtrix_engine.localai.llamacpp import LlamaRuntime
        tmp = Path({str(tmp_path)!r})
        rt = LlamaRuntime(tmp / "rt", tmp / "logs", JobManager(),
                          server_cmd=[sys.executable, str(tmp / "fake_server.py")])
        rt.ensure(tmp / "m.gguf", timeout=30)
        print(rt._proc.pid, flush=True)
        time.sleep(120)
    """)
    engine = subprocess.Popen([sys.executable, "-c", engine_like], stdout=subprocess.PIPE, text=True)
    server_pid = int(engine.stdout.readline())
    assert pid_alive(server_pid)
    engine.kill()  # TerminateProcess / SIGKILL: no cleanup code runs in the engine
    engine.wait()
    deadline = time.time() + 10
    while pid_alive(server_pid) and time.time() < deadline:
        time.sleep(0.25)
    alive = pid_alive(server_pid)
    if alive:
        subprocess.run(["taskkill", "/F", "/PID", str(server_pid)] if sys.platform == "win32"
                       else ["kill", "-9", str(server_pid)], capture_output=True)
    assert not alive, "llama-server outlived the engine"


def test_runtime_reports_crash_with_log_tail(tmp_path):
    script = tmp_path / "crash.py"
    script.write_text("import sys; print('error: unknown model architecture'); sys.exit(3)")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"m")
    rt = LlamaRuntime(tmp_path / "runtime", tmp_path / "logs", JobManager(), server_cmd=[sys.executable, str(script)])
    with pytest.raises(RuntimeError, match="unknown model architecture"):
        rt.ensure(model, timeout=30)
