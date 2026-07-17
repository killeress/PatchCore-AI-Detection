from pathlib import Path


def test_install_patch_allows_slow_model_startup_health_checks():
    script = Path("install_patch.sh").read_text(encoding="utf-8")

    assert 'HEALTH_TIMEOUT_SECONDS="${CAPI_HEALTH_TIMEOUT_SECONDS:-120}"' in script
    assert 'for _ in $(seq 1 "$HEALTH_TIMEOUT_SECONDS"); do' in script


def test_apply_records_health_failure_and_rollback_reason():
    source = Path("capi_update_agent.py").read_text(encoding="utf-8")

    assert "health check failed; automatic rollback completed" in source
    assert "health check failed (install command exited 2)" in source


def test_update_notice_points_to_both_apply_logs():
    template = Path("templates/_update_notice.html").read_text(encoding="utf-8")

    assert "update/auto_update.log" in template
    assert "update/manual_apply.log" in template
