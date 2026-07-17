import json
from pathlib import Path
import shutil
import subprocess
import uuid
import zipfile


def _write_patch_zip(path, version):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("VERSION", f"{version}\n")
        zf.writestr("release_manifest.json", json.dumps({"version": version}))


def test_publish_manifest_writes_latest_json():
    from capi_update_agent import main

    work_dir = Path("deployment") / f"_test_auto_update_publish_{uuid.uuid4().hex}"
    package_dir = work_dir / "packages"
    output_dir = work_dir / "served"
    package_dir.mkdir(parents=True)
    package = package_dir / "patchcore_ai_patch_2099.01.02.1.zip"

    try:
        _write_patch_zip(package, "2099.01.02.1")

        rc = main([
            "publish",
            "--package",
            str(package),
            "--output-dir",
            str(output_dir),
        ])

        assert rc == 0
        copied = output_dir / package.name
        manifest = json.loads((output_dir / "latest.json").read_text(encoding="utf-8"))

        assert copied.exists()
        assert manifest["version"] == "2099.01.02.1"
        assert manifest["package"] == package.name
        assert len(manifest["sha256"]) == 64
        assert manifest["requires_restart"] is True
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_check_downloads_package_and_stages_without_install(monkeypatch):
    import capi_update_agent

    work_dir = Path("deployment") / f"_test_auto_update_check_{uuid.uuid4().hex}"
    app_root = work_dir / "app"
    source_dir = work_dir / "source"
    app_root.mkdir(parents=True)
    source_dir.mkdir()

    try:
        (app_root / "VERSION").write_text("2099.01.02.0\n", encoding="utf-8")

        package = source_dir / "patchcore_ai_patch_2099.01.02.1.zip"
        _write_patch_zip(package, "2099.01.02.1")
        manifest = {
            "version": "2099.01.02.1",
            "package": package.name,
            "sha256": capi_update_agent._sha256_file(package),
        }
        manifest_path = source_dir / "latest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        def fail_run(*args, **kwargs):
            raise AssertionError("check must not run the installer")

        monkeypatch.setattr(capi_update_agent.subprocess, "run", fail_run)

        rc = capi_update_agent.main([
            "check",
            "--manifest-url",
            str(manifest_path),
            "--app-root",
            str(app_root),
            "--health-url",
            "http://127.0.0.1/api/version",
        ])

        state = json.loads((app_root / "update" / "auto_update_state.json").read_text(encoding="utf-8"))
        downloaded = app_root / "update" / "incoming" / package.name

        assert rc == 0
        assert downloaded.exists()
        assert state["status"] == "pending"
        assert state["pending_update"]["version"] == "2099.01.02.1"
        assert Path(state["pending_update"]["package"]) == downloaded.resolve()
        assert state["pending_update"]["health_url"] == "http://127.0.0.1/api/version"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_apply_staged_update_runs_installer_and_records_success(monkeypatch):
    import capi_update_agent

    work_dir = Path("deployment") / f"_test_auto_update_apply_{uuid.uuid4().hex}"
    app_root = work_dir / "app"
    incoming = app_root / "update" / "incoming"
    incoming.mkdir(parents=True)

    try:
        (app_root / "VERSION").write_text("2099.01.02.0\n", encoding="utf-8")
        package = incoming / "patchcore_ai_patch_2099.01.02.1.zip"
        _write_patch_zip(package, "2099.01.02.1")
        state_file = app_root / "update" / "auto_update_state.json"
        state_file.write_text(json.dumps({
            "status": "pending",
            "pending_update": {
                "version": "2099.01.02.1",
                "package": str(package.resolve()),
                "sha256": capi_update_agent._sha256_file(package),
                "manifest_url": "http://updates/latest.json",
                "health_url": "http://127.0.0.1/api/version",
            },
        }), encoding="utf-8")

        calls = []

        def fake_run(cmd, cwd, text, capture_output, env):
            calls.append({"cmd": cmd, "cwd": cwd, "env": env})
            (app_root / "VERSION").write_text("2099.01.02.1\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="installed\n", stderr="")

        monkeypatch.setattr(capi_update_agent.subprocess, "run", fake_run)

        rc = capi_update_agent.main([
            "apply",
            "--app-root",
            str(app_root),
            "--install-command",
            "fake-install",
        ])

        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert rc == 0
        assert calls[0]["cmd"][0] == "fake-install"
        assert Path(calls[0]["cmd"][1]) == package.resolve()
        assert calls[0]["env"]["CAPI_PATCH_AUTO_ROLLBACK"] == "1"
        assert calls[0]["env"]["CAPI_HEALTH_URL"] == "http://127.0.0.1/api/version"
        assert calls[0]["env"]["CAPI_PYTHON_BIN"] == capi_update_agent.sys.executable
        assert state["status"] == "current"
        assert state["last_success"]["version"] == "2099.01.02.1"
        assert "pending_update" not in state
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_check_skips_when_version_is_current(monkeypatch):
    import capi_update_agent

    work_dir = Path("deployment") / f"_test_auto_update_skip_{uuid.uuid4().hex}"
    app_root = work_dir / "app"
    source_dir = work_dir / "source"
    app_root.mkdir(parents=True)
    source_dir.mkdir()

    try:
        (app_root / "VERSION").write_text("2099.01.02.1\n", encoding="utf-8")

        manifest_path = source_dir / "latest.json"
        manifest_path.write_text(
            json.dumps({
                "version": "2099.01.02.1",
                "package": "patchcore_ai_patch_2099.01.02.1.zip",
                "sha256": "0" * 64,
            }),
            encoding="utf-8",
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("installer should not run")

        monkeypatch.setattr(capi_update_agent.subprocess, "run", fail_run)

        rc = capi_update_agent.main([
            "check",
            "--manifest-url",
            str(manifest_path),
            "--app-root",
            str(app_root),
        ])

        assert rc == 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
