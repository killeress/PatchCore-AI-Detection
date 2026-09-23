"""Exercise the publish-server shell path without installing or restarting AI."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _bash():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if os.name == "nt" and git_bash.is_file():
        return str(git_bash)
    executable = shutil.which("bash")
    if not executable:
        pytest.skip("bash is not installed")
    return executable


def _run(tmp_path, scenario, timeout="120", manifest=True):
    app = tmp_path / "app"
    app.mkdir()
    script = app / "promote_update.sh"
    script.write_text(Path("promote_update.sh").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    if manifest:
        (repo / "latest.json").write_text('{"version":"2026.09.23.1"}', encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    commands = {
        "python3": '''#!/bin/bash
echo started >> "$TEST_STARTS"
[ "$1" = "-u" ] || exit 90
if [ "$TEST_SCENARIO" = "exited" ]; then
    echo "test bind failure" >&2
    exit 3
fi
exec sleep 60
''',
        "curl": '''#!/bin/bash
[ "$1" = "-q" ] && [ "$2" = "--noproxy" ] && [ "$3" = "*" ] || exit 91
[[ "$*" == *"--connect-timeout 2"* ]] || exit 92
[[ "$*" == *"--max-time "* ]] || exit 93
count=0
if [ -f "$TEST_PROBES" ]; then read -r count < "$TEST_PROBES"; fi
count=$((count + 1))
echo "$count" > "$TEST_PROBES"
if [ "$TEST_SCENARIO" = "existing" ] || { [ "$TEST_SCENARIO" = "slow" ] && [ "$count" -ge 13 ]; }; then
    echo '{"version":"2026.09.23.1"}'
    exit 0
fi
echo 'curl: (7) test connection refused' >&2
exit 7
''',
    }
    for name, content in commands.items():
        path = bin_dir / name
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    env = os.environ.copy()
    env.update(CAPI_UPDATE_REPO=repo.as_posix(),
               CAPI_UPDATE_HTTP_LOG=(tmp_path / "http.log").as_posix(),
               CAPI_UPDATE_HTTP_START_TIMEOUT_SECONDS=timeout,
               TEST_SCENARIO=scenario,
               TEST_STARTS=(tmp_path / "starts").as_posix(),
               TEST_PROBES=(tmp_path / "probes").as_posix())
    # Source in a wrapper so EXIT also cleans up a successfully started child.
    wrapper = '''test_bin="$1"
if command -v cygpath >/dev/null 2>&1; then test_bin="$(cygpath -u "$test_bin")"; fi
export PATH="$test_bin:$PATH"
trap 'for pid in $(jobs -pr); do kill "$pid" 2>/dev/null || true; done' EXIT
source "$2" --serve-only
'''
    result = subprocess.run([_bash(), "-c", wrapper, script.as_posix(), bin_dir.as_posix(), script.as_posix()],
                            env=env, text=True, capture_output=True, timeout=30)
    return result


def test_server_can_start_after_old_ten_second_window_without_reinstall(tmp_path):
    result = _run(tmp_path, "slow")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "timeout=120s" in result.stdout
    assert "HTTP server started" in result.stdout
    assert "Skipping install" in result.stdout
    assert "Keeping existing published package metadata" in result.stdout
    assert "Promote completed" in result.stdout
    assert int((tmp_path / "probes").read_text()) >= 14
    assert (tmp_path / "starts").read_text().splitlines() == ["started"]


def test_existing_server_is_reused_without_starting_another(tmp_path):
    result = _run(tmp_path, "existing")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "already serves latest.json" in result.stdout
    assert not (tmp_path / "starts").exists()


def test_timeout_reports_probe_error_and_keeps_published_metadata(tmp_path):
    result = _run(tmp_path, "refused", timeout="2")
    assert result.returncode == 1
    assert "Last local HTTP probe: curl: (7) test connection refused" in result.stdout
    assert "stopping it" in result.stdout
    assert "Retry HTTP only:" in result.stdout
    assert (tmp_path / "repo/latest.json").is_file()


def test_process_exit_reports_startup_log(tmp_path):
    result = _run(tmp_path, "exited")
    assert result.returncode == 1
    assert "process exited during startup" in result.stdout
    assert "test bind failure" in result.stdout


@pytest.mark.parametrize("timeout", ["0", "-1", "bad"])
def test_invalid_timeout_fails_before_starting_any_service(tmp_path, timeout):
    result = _run(tmp_path, "existing", timeout=timeout)
    assert result.returncode == 1
    assert "must be a positive integer" in result.stdout
    assert not (tmp_path / "starts").exists()
    assert not (tmp_path / "probes").exists()


def test_serve_only_requires_published_manifest(tmp_path):
    result = _run(tmp_path, "existing", manifest=False)
    assert result.returncode == 1
    assert "requires existing" in result.stdout
    assert not (tmp_path / "starts").exists()
