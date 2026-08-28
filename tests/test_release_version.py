import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import pytest


def test_api_version_handler_returns_release_info(monkeypatch):
    import capi_web
    from capi_web import CAPIWebHandler

    captured = {}
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler._send_json = lambda payload, status=200: captured.update({"payload": payload, "status": status})

    monkeypatch.setattr(capi_web, "get_version_info", lambda: {"version": "2099.01.02.3"})

    handler._handle_api_version()

    assert captured == {"payload": {"version": "2099.01.02.3"}, "status": 200}


def test_release_notes_markdown_renders_basic_html():
    from capi_web import CAPIWebHandler

    rendered = CAPIWebHandler._render_changelog_html(
        "# 更新紀錄\n\n## 2026.06.29.1\n\n### RIC 報表\n\n- 新增規格內明細\n"
    )

    assert "<h1>更新紀錄</h1>" in rendered
    assert "<h2>2026.06.29.1</h2>" in rendered
    assert "<h3>RIC 報表</h3>" in rendered
    assert "<li>新增規格內明細</li>" in rendered


def test_install_patch_default_health_url_matches_production_port():
    script = Path("install_patch.sh").read_text(encoding="utf-8")

    assert 'HEALTH_URL="${CAPI_HEALTH_URL:-http://127.0.0.1/api/version}"' in script
    assert "127.0.0.1:8080/api/version" not in script


def test_install_patch_applies_bundled_mark_worker_after_main_health_check():
    script = Path("install_patch.sh").read_text(encoding="utf-8")

    main_health = script.index('echo "[5/6] Health check..."')
    worker_install = script.index('echo "[6/6] Applying bundled MARK shadow worker..."')

    assert main_health < worker_install
    assert 'MARK_SHADOW_TARGET_ROOT="${MARK_SHADOW_TARGET_ROOT:-/aidata/capi_ai/mark_shadow}"' in script
    assert 'elif [ ! -L "$MARK_SHADOW_TARGET_ROOT/current" ]; then' in script
    assert 'MARK_SHADOW_OUTER_CHECKSUMS_VERIFIED=1' in script
    assert 'WORKER_INSTALLER="$APP_ROOT/mark_shadow/install_worker_hotfix.sh"' in script


def test_worker_hotfix_installer_accepts_verified_codeonly_layout():
    script = Path("mark_shadow/install_worker_hotfix.sh").read_text(encoding="utf-8")

    assert 'WORKER_SOURCE="$SCRIPT_DIR/paddle_shadow_worker.py"' in script
    assert 'MARK_SHADOW_OUTER_CHECKSUMS_VERIFIED:-0' in script
    assert 'WORKER_SOURCE="$PATCH_ROOT/worker/paddle_shadow_worker.py"' in script
    assert 'CHECKSUM_FILE="$PATCH_ROOT/SHA256SUMS"' in script


def test_promote_update_verifies_background_http_server_startup():
    script = Path("promote_update.sh").read_text(encoding="utf-8")

    assert 'PACKAGE_DIR="${CAPI_UPDATE_PACKAGE_DIR:-$UPDATE_REPO/staging}"' in script
    assert "patchcore_ai_release_*_codeonly.zip" in script
    assert "sort -V" in script
    assert 'Usage: $0 [release-zip]' in script
    assert '--directory "$UPDATE_REPO"' in script
    assert "HTTP_PID=$!" in script
    assert 'if ! kill -0 "$HTTP_PID"' in script
    assert 'tail -n 50 "$HTTP_LOG"' in script
    assert 'sleep 2' not in script


def test_start_server_prefers_capi_python_before_system_python():
    script = Path("start_server.sh").read_text(encoding="utf-8")

    assert 'PYTHON="${CAPI_PYTHON_BIN:-}"' in script
    assert "/opt/miniconda3/envs/CAPI-PC/bin/python3" in script
    assert "PYTHON=$(command -v python3 || command -v python || true)" in script


def test_start_server_handles_zombies_and_checks_configured_ports():
    script = Path("start_server.sh").read_text(encoding="utf-8")

    assert "pid_is_zombie" in script
    assert "while [ $i -lt 10 ] && pid_is_server \"$pid\"; do" in script
    assert "check_configured_ports" in script
    assert "wait_for_configured_ports" in script
    assert "PORT_WAIT_SECONDS=10" in script
    assert "setsid" in script
    assert 'web.get("port", 8080)' in script
    assert 'server.get("port", 7907)' in script
    assert "ss" in script or "lsof" in script


@pytest.mark.parametrize(
    ("hostname", "station_name"),
    [("CAPI07", "CAPI"), ("mod2-aapi09", "AAPI")],
)
def test_base_template_uses_hostname_for_station_brand(
    monkeypatch,
    hostname,
    station_name,
):
    import capi_web
    from capi_web import CAPIWebHandler

    original_env = CAPIWebHandler.jinja_env
    monkeypatch.setattr(capi_web.socket, "gethostname", lambda: hostname)
    CAPIWebHandler.jinja_env = None
    try:
        CAPIWebHandler.init_jinja()
        template = CAPIWebHandler.jinja_env.get_template("base.html")
        rendered = template.render(request_path="/")
        dashboard_rendered = CAPIWebHandler.jinja_env.get_template(
            "dashboard_v3.html"
        ).render()
    finally:
        CAPIWebHandler.jinja_env = original_env

    assert f"<title>[{hostname}] {station_name} AI 推論伺服器</title>" in rendered
    assert f"{station_name} AD-AI智能檢測" in rendered
    assert f'class="host-identity-badge" title="{hostname}">{hostname}</span>' in rendered
    assert f"主機：{hostname} ｜ 版本：" in rendered
    assert f"MOD2 {station_name} AD - 即時監控儀表板" in dashboard_rendered
    assert f"{station_name} AI Vision" in dashboard_rendered
    if station_name == "AAPI":
        assert "CAPI AD-AI智能檢測" not in rendered
        assert "CAPI AI Vision" not in dashboard_rendered


def test_start_server_banner_uses_hostname_station(monkeypatch):
    import start_server

    monkeypatch.setattr(start_server.socket, "gethostname", lambda: "mod2-aapi09")

    assert start_server._station_name() == "AAPI"


def test_build_release_zip_includes_manifest_checksums_and_excludes_static_dirs(tmp_path, monkeypatch):
    from scripts import build_deploy_zip

    version = "2099.01.02.3"
    output_dir = tmp_path / "release"
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [])

    rc = build_deploy_zip.main([
        "--no-backbone",
        "--version",
        version,
        "--output-dir",
        str(output_dir),
    ])

    assert rc == 0
    zip_path = output_dir / f"patchcore_ai_release_{version}_codeonly.zip"
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "VERSION" in names
        assert "CHANGELOG.md" in names
        assert "release_manifest.json" in names
        assert "checksums.txt" in names
        assert "capi_version.py" in names
        assert "capi_station_adapter.py" in names
        assert "start_server.py" in names
        assert "capi_update_agent.py" in names
        assert "promote_update.sh" in names
        assert "setup_auto_update_client.sh" in names
        assert "templates/release_notes.html" in names
        assert "server_config_patch.yaml.example" in names
        assert "templates/_update_notice.html" in names
        assert "templates/dashboard_v3.html" in names
        assert "templates/debug_inference.html" in names
        assert "templates/record_detail.html" in names
        assert "templates/record_detail_v3.html" in names
        assert "templates/ric_report.html" in names
        assert "templates/train_new/step1_scope.html" in names
        assert "capi_edge_cv.py" in names
        assert "capi_heatmap.py" in names
        assert "capi_heatmap_diagnostics.py" in names
        assert "capi_grid_canonicalization.py" in names
        assert "capi_image_orientation.py" in names
        assert "capi_image_preprocess_lab.py" in names
        assert "capi_dataset_export.py" in names
        assert "capi_mark_calibration.py" in names
        assert "capi_mark_detector.py" in names
        assert "capi_mark_shadow.py" in names
        assert "mark_shadow/paddle_shadow_worker.py" in names
        assert "mark_shadow/install_worker_hotfix.sh" in names
        assert "mark_shadow/README_WORKER_HOTFIX.txt" in names
        assert "capi_model_validation.py" in names
        assert "configs/mes_defect_codes.json" in names
        assert "capi_patchcore_feature_cleaning.py" in names
        assert "capi_scratch_batch.py" in names
        assert "capi_scratch_export.py" in names
        assert "central_dashboard/index.html" in names
        assert "central_dashboard/app.js" in names
        assert "central_dashboard/config.js" in names
        assert "central_dashboard/styles.css" in names
        assert "central_dashboard/settings.html" in names
        assert "central_dashboard/README.md" in names
        assert "capi_mes_credentials.py" not in names
        assert "scripts/over_review_poc/train_final_model.py" in names
        assert "scratch_classifier.py" in names
        assert "scratch_filter.py" in names
        for bundled_worker_file in (
            "mark_shadow/paddle_shadow_worker.py",
            "mark_shadow/install_worker_hotfix.sh",
            "mark_shadow/README_WORKER_HOTFIX.txt",
        ):
            assert zf.read(bundled_worker_file) == Path(bundled_worker_file).read_bytes()

        managed_assets = {
            rel
            for rel in build_deploy_zip._git_file_list([
                "ls-files", "--cached", "--others", "--exclude-standard", "--", "templates", "static"
            ])
            if (
                (build_deploy_zip.PROJECT_ROOT / rel).is_file()
                and not build_deploy_zip._is_codeonly_excluded_file(rel)
            )
        }
        assert managed_assets <= names
        assert not any(name.startswith("templates/imgs/") for name in names)
        assert not any(name.startswith("static/") for name in names)

        web_source = Path("capi_web.py").read_text(encoding="utf-8")
        direct_templates = set(re.findall(r'get_template\(\s*["\']([^"\']+)["\']', web_source))
        assert direct_templates <= {
            name.removeprefix("templates/") for name in names if name.startswith("templates/")
        }

        assert zf.read("VERSION").decode("utf-8").strip() == version
        manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))
        checksums = zf.read("checksums.txt").decode("utf-8")
        readme = zf.read("README.txt").decode("utf-8")
        for item in manifest["files"]:
            data = zf.read(item["path"])
            assert len(data) == item["size_bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]

    assert manifest["version"] == version
    assert manifest["package_type"] == "codeonly"
    assert manifest["contains_local_credentials"] is False
    assert manifest["requires_restart"] is True
    assert manifest["git_commit"] == build_deploy_zip._git_commit()
    assert manifest["git_commit_full"] == build_deploy_zip._git_commit_full()
    assert manifest["git_worktree_dirty"] is False
    assert manifest["git_dirty"] is False
    assert manifest["git_dirty_files"] == []
    assert manifest["source_mode"] == "git_commit"
    canonical_files = json.dumps(
        manifest["files"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert manifest["content_tree_sha256"] == hashlib.sha256(canonical_files).hexdigest()
    assert any(item["path"] == "capi_mark_calibration.py" for item in manifest["files"])
    assert any(item["path"] == "capi_mark_shadow.py" for item in manifest["files"])
    assert any(item["path"] == "mark_shadow/paddle_shadow_worker.py" for item in manifest["files"])
    assert any(item["path"] == "mark_shadow/install_worker_hotfix.sh" for item in manifest["files"])
    assert any(item["path"] == "capi_station_adapter.py" for item in manifest["files"])
    assert any(item["path"] == "capi_grid_canonicalization.py" for item in manifest["files"])
    assert any(item["path"] == "start_server.py" for item in manifest["files"])
    assert any(item["path"] == "capi_web.py" for item in manifest["files"])
    assert not any(item["path"] == "capi_mes_credentials.py" for item in manifest["files"])
    assert "  capi_mark_calibration.py\n" in checksums
    assert "  capi_mark_shadow.py\n" in checksums
    assert "  mark_shadow/paddle_shadow_worker.py\n" in checksums
    assert "  mark_shadow/install_worker_hotfix.sh\n" in checksums
    assert "  capi_station_adapter.py\n" in checksums
    assert "  capi_grid_canonicalization.py\n" in checksums
    assert "  start_server.py\n" in checksums
    assert "  capi_web.py\n" in checksums
    assert "  capi_mes_credentials.py\n" not in checksums
    assert "unzip -o /path/to/patchcore_ai_release_<version>_codeonly.zip install_patch.sh" in readme
    assert "只手動解壓並重啟主程式，不會更新" in readme


def test_codeonly_includes_local_credentials_only_with_explicit_flag(
    tmp_path, monkeypatch, capsys,
):
    from scripts import build_deploy_zip

    version = "2099.01.02.39"
    output_dir = tmp_path / "release-with-credentials"
    monkeypatch.setattr(build_deploy_zip, "CODE_FILES", ["capi_version.py"])
    monkeypatch.setattr(build_deploy_zip, "_git_managed_asset_files", lambda: [])
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [])

    assert build_deploy_zip.main([
        "--no-backbone",
        "--allow-dirty",
        "--include-local-credentials",
        "--version",
        version,
        "--output-dir",
        str(output_dir),
    ]) == 0

    output = capsys.readouterr().out
    assert "including plaintext local MES credentials" in output
    with zipfile.ZipFile(
        output_dir / f"patchcore_ai_release_{version}_codeonly.zip"
    ) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))

    assert "capi_mes_credentials.py" in names
    assert manifest["contains_local_credentials"] is True
    assert manifest["git_dirty_files"] == ["capi_mes_credentials.py"]
    assert manifest["source_mode"] == "working_tree"


def test_codeonly_warns_when_excluded_static_assets_change(tmp_path, monkeypatch, capsys):
    from scripts import build_deploy_zip

    version = "2099.01.02.38"
    output_dir = tmp_path / "codeonly-static-warning"
    monkeypatch.setattr(build_deploy_zip, "CODE_FILES", ["capi_version.py"])
    monkeypatch.setattr(build_deploy_zip, "_git_managed_asset_files", lambda: [
        "templates/base.html",
        "templates/imgs/new-banner.png",
        "static/js/new-widget.js",
    ])
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [
        "templates/imgs/new-banner.png",
        "static/js/new-widget.js",
    ])

    assert build_deploy_zip.main([
        "--no-backbone",
        "--version",
        version,
        "--output-dir",
        str(output_dir),
    ]) == 0

    output = capsys.readouterr().out
    assert "WARNING: excluded static assets changed" in output
    assert "templates/imgs/new-banner.png" in output
    assert "static/js/new-widget.js" in output

    with zipfile.ZipFile(output_dir / f"patchcore_ai_release_{version}_codeonly.zip") as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))

    assert "templates/base.html" in names
    assert "templates/imgs/new-banner.png" not in names
    assert "static/js/new-widget.js" not in names
    assert manifest["git_worktree_dirty"] is True
    assert manifest["git_dirty"] is False
    assert manifest["git_dirty_files"] == []


def test_build_release_zip_rejects_relevant_dirty_files_unless_overridden(tmp_path, monkeypatch):
    from scripts import build_deploy_zip

    version = "2099.01.02.30"
    monkeypatch.setattr(build_deploy_zip, "CODE_FILES", ["capi_version.py", "capi_web.py"])
    monkeypatch.setattr(build_deploy_zip, "_git_managed_asset_files", lambda: [])
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [
        "W0F00000_084556.tif",
        "outputs/report.pptx",
        "capi_web.py",
    ])

    with pytest.raises(RuntimeError, match=r"capi_web\.py.*--allow-dirty"):
        build_deploy_zip.main([
            "--no-backbone",
            "--version",
            version,
            "--output-dir",
            str(tmp_path / "rejected"),
        ])

    output_dir = tmp_path / "allowed"
    assert build_deploy_zip.main([
        "--no-backbone",
        "--allow-dirty",
        "--version",
        version,
        "--output-dir",
        str(output_dir),
    ]) == 0

    zip_path = output_dir / f"patchcore_ai_release_{version}_codeonly.zip"
    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))

    assert manifest["git_worktree_dirty"] is True
    assert manifest["git_dirty"] is True
    assert manifest["git_dirty_files"] == ["capi_web.py"]
    assert manifest["source_mode"] == "working_tree"


def test_build_release_zip_ignores_unrelated_dirty_files(tmp_path, monkeypatch):
    from scripts import build_deploy_zip

    version = "2099.01.02.31"
    output_dir = tmp_path / "unrelated-dirty"
    monkeypatch.setattr(build_deploy_zip, "CODE_FILES", ["capi_version.py"])
    monkeypatch.setattr(build_deploy_zip, "_git_managed_asset_files", lambda: [])
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [
        "W0F00000_084556.tif",
        "outputs/report.pptx",
    ])

    assert build_deploy_zip.main([
        "--no-backbone",
        "--version",
        version,
        "--output-dir",
        str(output_dir),
    ]) == 0

    zip_path = output_dir / f"patchcore_ai_release_{version}_codeonly.zip"
    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))

    assert manifest["git_worktree_dirty"] is True
    assert manifest["git_dirty"] is False
    assert manifest["git_dirty_files"] == []
    assert manifest["source_mode"] == "git_commit"


def test_build_release_zip_fails_before_writing_when_required_code_file_is_missing(tmp_path, monkeypatch):
    from scripts import build_deploy_zip

    output_dir = tmp_path / "missing"
    monkeypatch.setattr(build_deploy_zip, "CODE_FILES", ["capi_version.py", "missing_required_module.py"])
    monkeypatch.setattr(build_deploy_zip, "_git_managed_asset_files", lambda: [])
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [])

    with pytest.raises(FileNotFoundError, match="missing_required_module.py"):
        build_deploy_zip.main([
            "--no-backbone",
            "--version",
            "2099.01.02.32",
            "--output-dir",
            str(output_dir),
        ])

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("mode_args", "failed_query"),
    [
        (["--no-backbone"], "changed-files"),
        (["--patch-only"], "changed-files"),
        (["--no-backbone"], "managed-assets"),
        (["--no-backbone"], "head"),
        (["--patch-only"], "head"),
    ],
)
def test_build_zip_fails_closed_before_output_when_git_query_fails(
    tmp_path, monkeypatch, mode_args, failed_query
):
    from scripts import build_deploy_zip

    output_dir = tmp_path / failed_query
    real_run = subprocess.run
    monkeypatch.setattr(build_deploy_zip, "CODE_FILES", ["capi_version.py"])

    if failed_query != "changed-files":
        monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [])
    if failed_query != "managed-assets":
        monkeypatch.setattr(build_deploy_zip, "_git_managed_asset_files", lambda: [])

    def fail_selected_git_query(command, *args, **kwargs):
        should_fail = (
            (failed_query == "changed-files" and "diff" in command)
            or (failed_query == "managed-assets" and "--cached" in command)
            or (failed_query == "head" and "rev-parse" in command)
        )
        if should_fail:
            raise subprocess.CalledProcessError(128, command, stderr="fatal: test git failure")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(build_deploy_zip.subprocess, "run", fail_selected_git_query)

    with pytest.raises(RuntimeError, match=r"git .* failed.*test git failure"):
        build_deploy_zip.main([
            *mode_args,
            "--version",
            "2099.01.02.33",
            "--output-dir",
            str(output_dir),
        ])

    assert not output_dir.exists()


def test_codeonly_dirty_gate_ignores_backbone_but_full_release_does_not(tmp_path, monkeypatch):
    from scripts import build_deploy_zip

    changed_backbone = "deployment/torch_hub_cache/models/backbone.bin"
    monkeypatch.setattr(build_deploy_zip, "CODE_FILES", ["capi_version.py"])
    monkeypatch.setattr(build_deploy_zip, "_git_managed_asset_files", lambda: [])
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [changed_backbone])

    codeonly_output = tmp_path / "codeonly"
    assert build_deploy_zip.main([
        "--no-backbone",
        "--version",
        "2099.01.02.34",
        "--output-dir",
        str(codeonly_output),
    ]) == 0

    with zipfile.ZipFile(codeonly_output / "patchcore_ai_release_2099.01.02.34_codeonly.zip") as zf:
        manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))
    assert manifest["git_worktree_dirty"] is True
    assert manifest["git_dirty"] is False
    assert manifest["git_dirty_files"] == []

    full_output = tmp_path / "full"
    with pytest.raises(RuntimeError, match="torch_hub_cache"):
        build_deploy_zip.main([
            "--version",
            "2099.01.02.35",
            "--output-dir",
            str(full_output),
        ])
    assert not full_output.exists()


def test_release_dirty_gate_ignores_unlisted_root_python_file(tmp_path, monkeypatch):
    from scripts import build_deploy_zip

    output_dir = tmp_path / "unlisted-root-python"
    monkeypatch.setattr(build_deploy_zip, "CODE_FILES", ["capi_version.py"])
    monkeypatch.setattr(build_deploy_zip, "_git_managed_asset_files", lambda: [])
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: ["experimental_module.py"])

    assert build_deploy_zip.main([
        "--no-backbone",
        "--version",
        "2099.01.02.36",
        "--output-dir",
        str(output_dir),
    ]) == 0

    with zipfile.ZipFile(output_dir / "patchcore_ai_release_2099.01.02.36_codeonly.zip") as zf:
        manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))
    assert manifest["git_worktree_dirty"] is True
    assert manifest["git_dirty"] is False
    assert manifest["git_dirty_files"] == []


def test_release_dirty_gate_keeps_deleted_web_asset_relevant(tmp_path, monkeypatch):
    from scripts import build_deploy_zip

    output_dir = tmp_path / "deleted-web-asset"
    monkeypatch.setattr(build_deploy_zip, "CODE_FILES", ["capi_version.py"])
    monkeypatch.setattr(build_deploy_zip, "_git_managed_asset_files", lambda: [])
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: ["templates/deleted_page.html"])

    with pytest.raises(RuntimeError, match="templates/deleted_page.html"):
        build_deploy_zip.main([
            "--no-backbone",
            "--version",
            "2099.01.02.37",
            "--output-dir",
            str(output_dir),
        ])

    assert not output_dir.exists()


def test_add_file_records_actual_zip_entry_bytes_when_source_changes(tmp_path, monkeypatch):
    from scripts import build_deploy_zip

    source = tmp_path / "source.sh"
    zip_path = tmp_path / "race.zip"
    original_data = b"#!/bin/bash\necho original\n"
    source.write_bytes(original_data)
    original_write = zipfile.ZipFile.write

    def write_then_change_source(zf, filename, arcname=None, *args, **kwargs):
        result = original_write(zf, filename, arcname, *args, **kwargs)
        Path(filename).write_bytes(b"changed after zip write")
        return result

    monkeypatch.setattr(zipfile.ZipFile, "write", write_then_change_source)
    entries = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        build_deploy_zip._add_file(zf, source, "source.sh", entries)

    with zipfile.ZipFile(zip_path) as zf:
        packed_data = zf.read("source.sh")
    assert packed_data == original_data
    assert entries == [{
        "path": "source.sh",
        "size_bytes": len(packed_data),
        "sha256": hashlib.sha256(packed_data).hexdigest(),
    }]


def test_build_patch_zip_includes_only_deployable_changes(tmp_path, monkeypatch):
    from scripts import build_deploy_zip

    version = "2099.01.02.4"
    output_dir = tmp_path / "patch"
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [
        "capi_web.py",
        "capi_update_agent.py",
        "templates/base.html",
        "static/favicon.svg",
        "mark_shadow/paddle_shadow_worker.py",
        "tests/test_release_version.py",
        "scripts/build_deploy_zip.py",
        "Sample/ignore.py",
    ])

    rc = build_deploy_zip.main([
        "--patch-only",
        "--version",
        version,
        "--output-dir",
        str(output_dir),
    ])

    assert rc == 0
    zip_path = output_dir / f"patchcore_ai_patch_{version}.zip"
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))
        scripts = {
            script_name: zf.read(script_name)
            for script_name in (
                "start_server.sh",
                "install_patch.sh",
                "rollback_patch.sh",
                "promote_update.sh",
                "setup_auto_update_client.sh",
            )
        }

    assert manifest["package_type"] == "patch"
    assert manifest["source_mode"] == "working_tree"
    assert manifest["git_worktree_dirty"] is True
    assert manifest["git_dirty"] is True
    assert "tests/test_release_version.py" not in manifest["git_dirty_files"]
    assert "capi_web.py" in names
    assert "capi_update_agent.py" in names
    assert "templates/base.html" in names
    assert "static/favicon.svg" in names
    assert "mark_shadow/paddle_shadow_worker.py" in names
    assert "install_patch.sh" in names
    assert "rollback_patch.sh" in names
    assert "start_server.sh" in names
    assert "promote_update.sh" in names
    assert "setup_auto_update_client.sh" in names
    assert "VERSION" in names
    assert "CHANGELOG.md" in names
    assert "tests/test_release_version.py" not in names
    assert "scripts/build_deploy_zip.py" not in names
    assert "Sample/ignore.py" not in names
    assert "capi_config.py" not in names

    for data in scripts.values():
        assert data.startswith(b"#!/bin/bash\n")
        assert b"\r\n" not in data
