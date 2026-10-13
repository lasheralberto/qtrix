import os
import subprocess
import sys
import time

from qtrix_engine.__main__ import pid_alive


def test_pid_alive_for_running_and_finished_processes():
    assert pid_alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not pid_alive(child.pid)


def test_engine_exits_when_parent_dies(tmp_path):
    """An engine whose desktop shell vanished must stop by itself instead of keeping the DB locked."""
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    env = {**os.environ, "QTRIX_HOME": str(tmp_path), "QTRIX_PARENT_PID": str(parent.pid)}
    engine = subprocess.Popen([sys.executable, "-m", "qtrix_engine", "--port", "0"], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert '"ready"' in engine.stdout.readline()
        parent.kill()
        parent.wait()
        deadline = time.time() + 20
        while engine.poll() is None and time.time() < deadline:
            time.sleep(0.5)
        assert engine.poll() == 0, "engine kept running after its parent died"
    finally:
        engine.kill()
        parent.kill()
