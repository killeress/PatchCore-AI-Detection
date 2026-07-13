import json
from pathlib import Path
import shutil
import uuid
import zipfile


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


def test_start_server_prefers_capi_python_before_system_python():
    script = Path("start_server.sh").read_text(encoding="utf-8")

    assert 'PYTHON="${CAPI_PYTHON_BIN:-}"' in script
    assert "/opt/miniconda3/envs/CAPI-PC/bin/python3" in script
    assert "PYTHON=$(command -v python3 || command -v python || true)" in script


def test_base_template_includes_hostname_in_title_header_and_footer(monkeypatch):
    import capi_web
    from capi_web import CAPIWebHandler

    original_env = CAPIWebHandler.jinja_env
    monkeypatch.setattr(capi_web.socket, "gethostname", lambda: "CAPI07")
    CAPIWebHandler.jinja_env = None
    try:
        CAPIWebHandler.init_jinja()
        template = CAPIWebHandler.jinja_env.get_template("base.html")
        rendered = template.render(request_path="/")
    finally:
        CAPIWebHandler.jinja_env = original_env

    assert "<title>[CAPI07] CAPI AI 推論伺服器</title>" in rendered
    assert 'class="host-identity-badge" title="CAPI07">CAPI07</span>' in rendered
    assert "主機：CAPI07 ｜ 版本：" in rendered


def test_build_release_zip_includes_manifest_and_checksums():
    from scripts import build_deploy_zip

    version = "2099.01.02.3"
    output_dir = Path("deployment") / f"_test_release_{uuid.uuid4().hex}"

    try:
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
            assert "capi_update_agent.py" in names
            assert "promote_update.sh" in names
            assert "setup_auto_update_client.sh" in names
            assert "templates/release_notes.html" in names
            assert "templates/debug_inference.html" in names
            assert "templates/record_detail.html" in names
            assert "templates/record_detail_v3.html" in names
            assert "templates/ric_report.html" in names
            assert "capi_edge_cv.py" in names
            assert "capi_heatmap.py" in names
            assert "capi_image_preprocess_lab.py" in names
            assert "capi_dataset_export.py" in names
            assert "capi_mark_detector.py" in names
            assert "capi_scratch_batch.py" in names
            assert "capi_scratch_export.py" in names
            assert "scratch_classifier.py" in names
            assert "scratch_filter.py" in names

            assert zf.read("VERSION").decode("utf-8").strip() == version
            manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))
            checksums = zf.read("checksums.txt").decode("utf-8")

        assert manifest["version"] == version
        assert manifest["package_type"] == "codeonly"
        assert manifest["requires_restart"] is True
        assert any(item["path"] == "capi_web.py" for item in manifest["files"])
        assert "  capi_web.py\n" in checksums
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_build_patch_zip_includes_only_deployable_changes(monkeypatch):
    from scripts import build_deploy_zip

    version = "2099.01.02.4"
    output_dir = Path("deployment") / f"_test_patch_{uuid.uuid4().hex}"
    monkeypatch.setattr(build_deploy_zip, "_git_changed_files", lambda: [
        "capi_web.py",
        "capi_update_agent.py",
        "templates/base.html",
        "tests/test_release_version.py",
        "scripts/build_deploy_zip.py",
        "Sample/ignore.py",
    ])

    try:
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
        assert "capi_web.py" in names
        assert "capi_update_agent.py" in names
        assert "templates/base.html" in names
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
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
