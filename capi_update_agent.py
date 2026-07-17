#!/usr/bin/env python3
"""Pull-based updater for staging and manually applying CAPI AI patch ZIPs.

Periodic checks only download and verify a pending package.  Applying the
package is a separate command so an operator can choose a safe restart time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    line = f"{_now_iso()} {message}"
    print(line)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_location(location: str, timeout: int) -> bytes:
    parsed = urlparse(location)
    if parsed.scheme in {"http", "https", "ftp", "file"}:
        with urllib.request.urlopen(location, timeout=timeout) as response:
            return response.read()
    return Path(location).read_bytes()


def _load_json_location(location: str, timeout: int) -> dict:
    data = _read_location(location, timeout)
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    return payload


def _normalize_manifest(payload: dict) -> dict:
    version = str(payload.get("version") or "").strip()
    package = str(payload.get("package") or "").strip()
    sha256 = str(payload.get("sha256") or "").strip().lower()

    if not version:
        raise ValueError("manifest missing version")
    if not package:
        raise ValueError("manifest missing package")
    if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
        raise ValueError("manifest sha256 must be a lowercase SHA-256 hex digest")

    normalized = dict(payload)
    normalized["version"] = version
    normalized["package"] = package
    normalized["sha256"] = sha256
    return normalized


def _resolve_package_location(manifest_location: str, package_ref: str) -> str:
    if urlparse(package_ref).scheme:
        return package_ref

    manifest_parsed = urlparse(manifest_location)
    if manifest_parsed.scheme:
        return urljoin(manifest_location, package_ref)

    return str((Path(manifest_location).resolve().parent / package_ref).resolve())


def _package_filename(location: str) -> str:
    parsed = urlparse(location)
    name = Path(unquote(parsed.path if parsed.scheme else location)).name
    if not name:
        raise ValueError(f"cannot determine package filename from {location!r}")
    return name


def _download_package(location: str, target: Path, timeout: int) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    parsed = urlparse(location)
    if parsed.scheme in {"http", "https", "ftp", "file"}:
        with urllib.request.urlopen(location, timeout=timeout) as response:
            with tmp.open("wb") as f:
                shutil.copyfileobj(response, f)
    else:
        shutil.copy2(Path(location), tmp)
    os.replace(tmp, target)
    return target


def _read_current_version(app_root: Path) -> str:
    version_file = app_root / "VERSION"
    if not version_file.exists():
        return "unknown"
    return version_file.read_text(encoding="utf-8").strip() or "unknown"


def _load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {}
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_name(state_file.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, state_file)


def _read_zip_version(package: Path) -> str:
    with zipfile.ZipFile(package) as zf:
        names = set(zf.namelist())
        if "release_manifest.json" in names:
            manifest = json.loads(zf.read("release_manifest.json").decode("utf-8"))
            version = str(manifest.get("version") or "").strip()
            if version:
                return version
        if "VERSION" in names:
            version = zf.read("VERSION").decode("utf-8").strip()
            if version:
                return version
    raise ValueError(f"cannot find version in {package}")


def publish_manifest(args: argparse.Namespace) -> int:
    package = args.package.resolve()
    if not package.is_file():
        raise FileNotFoundError(package)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / package.name
    if destination != package:
        shutil.copy2(package, destination)

    version = args.version or _read_zip_version(destination)
    manifest = {
        "version": version,
        "package": args.package_url or destination.name,
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "published_at": _now_iso(),
        "requires_restart": True,
    }

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Package {destination}")
    return 0


def check_once(args: argparse.Namespace) -> int:
    app_root = args.app_root.resolve()
    update_dir = (app_root / "update").resolve()
    log_file = args.log_file or update_dir / "auto_update.log"
    state_file = args.state_file or update_dir / "auto_update_state.json"
    download_dir = args.download_dir or update_dir / "incoming"

    manifest = _normalize_manifest(_load_json_location(args.manifest_url, args.timeout))
    wanted_version = manifest["version"]
    current_version = _read_current_version(app_root)
    state = _load_state(state_file)

    if wanted_version == current_version:
        pending = state.get("pending_update")
        if isinstance(pending, dict) and pending.get("version") == wanted_version:
            state.pop("pending_update", None)
            state.pop("last_failed", None)
            state["status"] = "current"
            state["checked_at"] = _now_iso()
            _write_state(state_file, state)
        _append_log(log_file, f"already on version {wanted_version}; no update")
        return 0

    if state.get("status") in {"apply_requested", "installing"}:
        _append_log(log_file, "manual update is already in progress; skip this check")
        return 0

    package_location = _resolve_package_location(args.manifest_url, manifest["package"])
    package_path = download_dir / _package_filename(package_location)

    _append_log(log_file, f"update available {current_version} -> {wanted_version}")
    if args.dry_run:
        _append_log(log_file, f"dry run: would download {package_location}")
        return 0

    pending = state.get("pending_update") if isinstance(state.get("pending_update"), dict) else {}
    package_already_staged = (
        pending.get("version") == wanted_version
        and pending.get("sha256") == manifest["sha256"]
        and package_path.is_file()
        and _sha256_file(package_path) == manifest["sha256"]
    )
    if package_already_staged:
        _append_log(log_file, f"version {wanted_version} is already staged for manual apply")
    else:
        _append_log(log_file, f"downloading {package_location}")
        _download_package(package_location, package_path, args.timeout)

    actual_sha256 = _sha256_file(package_path)
    if actual_sha256 != manifest["sha256"]:
        state["status"] = "failed"
        state["last_failed"] = {
            "version": wanted_version,
            "at": _now_iso(),
            "reason": "checksum mismatch",
            "expected_sha256": manifest["sha256"],
            "actual_sha256": actual_sha256,
        }
        _write_state(state_file, state)
        raise RuntimeError(f"checksum mismatch for {package_path}")

    last_failed = state.get("last_failed") if isinstance(state.get("last_failed"), dict) else {}
    state["pending_update"] = {
        "version": wanted_version,
        "package": str(package_path.resolve()),
        "sha256": manifest["sha256"],
        "manifest_url": args.manifest_url,
        "detected_at": pending.get("detected_at") or _now_iso(),
        "health_url": args.health_url or pending.get("health_url"),
        "download_dir": str(Path(download_dir).resolve()),
    }
    state["checked_at"] = _now_iso()
    if last_failed.get("version") == wanted_version:
        state["status"] = "failed"
    else:
        state["status"] = "pending"
        state.pop("last_failed", None)
    _write_state(state_file, state)
    _append_log(log_file, f"staged version {wanted_version}; waiting for manual apply")
    return 0


def apply_pending(args: argparse.Namespace) -> int:
    app_root = args.app_root.resolve()
    update_dir = (app_root / "update").resolve()
    log_file = args.log_file or update_dir / "auto_update.log"
    state_file = args.state_file or update_dir / "auto_update_state.json"

    if args.delay > 0:
        _append_log(log_file, f"manual apply requested; starting in {args.delay}s")
        time.sleep(args.delay)

    state = _load_state(state_file)
    if state.get("status") == "installing":
        _append_log(log_file, "another manual update is already installing")
        return 5

    pending = state.get("pending_update")
    if not isinstance(pending, dict):
        _append_log(log_file, "no staged update to apply")
        return 3

    wanted_version = str(pending.get("version") or "").strip()
    expected_sha256 = str(pending.get("sha256") or "").strip().lower()
    download_dir = Path(
        args.download_dir or pending.get("download_dir") or update_dir / "incoming"
    ).resolve()
    current_version = _read_current_version(app_root)
    if wanted_version == current_version:
        state.pop("pending_update", None)
        state.pop("last_failed", None)
        state["status"] = "current"
        _write_state(state_file, state)
        _append_log(log_file, f"already on version {wanted_version}; cleared pending update")
        return 0

    try:
        package_path = Path(str(pending.get("package") or "")).resolve()
        try:
            package_path.relative_to(download_dir)
        except ValueError as exc:
            raise ValueError("staged package is outside update/incoming") from exc
        if not package_path.is_file():
            raise ValueError("staged package is missing")
        if len(expected_sha256) != 64:
            raise ValueError("pending update has invalid sha256")
        actual_sha256 = _sha256_file(package_path)
        if actual_sha256 != expected_sha256:
            raise ValueError("staged package checksum mismatch")
        if _read_zip_version(package_path) != wanted_version:
            raise ValueError("staged package version mismatch")
    except Exception as exc:
        state.pop("apply_requested", None)
        state["status"] = "failed"
        state["last_failed"] = {
            "version": wanted_version,
            "at": _now_iso(),
            "reason": str(exc),
        }
        _write_state(state_file, state)
        _append_log(log_file, f"manual apply validation failed: {exc}")
        return 4

    install_script = app_root / "install_patch.sh"
    if args.install_command:
        install_args = [args.install_command, str(package_path)]
    elif os.access(install_script, os.X_OK):
        install_args = [str(install_script), str(package_path)]
    else:
        install_args = ["bash", str(install_script), str(package_path)]
    env = os.environ.copy()
    if not args.no_auto_rollback:
        env["CAPI_PATCH_AUTO_ROLLBACK"] = "1"
    health_url = args.health_url or pending.get("health_url")
    if health_url:
        env["CAPI_HEALTH_URL"] = str(health_url)
    env.setdefault("CAPI_PYTHON_BIN", sys.executable)

    state.pop("apply_requested", None)
    state["status"] = "installing"
    state["installing"] = {
        "version": wanted_version,
        "started_at": _now_iso(),
    }
    _write_state(state_file, state)
    _append_log(log_file, f"manually installing {package_path}")
    proc = subprocess.run(
        install_args,
        cwd=app_root,
        text=True,
        capture_output=True,
        env=env,
    )
    if proc.stdout:
        _append_log(log_file, proc.stdout.rstrip())
    if proc.stderr:
        _append_log(log_file, proc.stderr.rstrip())

    if proc.returncode != 0:
        state.pop("installing", None)
        state["status"] = "failed"
        state["last_failed"] = {
            "version": wanted_version,
            "at": _now_iso(),
            "reason": f"install command exited {proc.returncode}",
            "package": str(package_path),
        }
        _write_state(state_file, state)
        return proc.returncode

    installed_version = _read_current_version(app_root)
    if installed_version != wanted_version:
        state.pop("installing", None)
        state["status"] = "failed"
        state["last_failed"] = {
            "version": wanted_version,
            "at": _now_iso(),
            "reason": f"installed VERSION is {installed_version}",
            "package": str(package_path),
        }
        _write_state(state_file, state)
        return 4

    state["last_success"] = {
        "version": wanted_version,
        "at": _now_iso(),
        "package": str(package_path),
        "manifest_url": pending.get("manifest_url"),
    }
    state.pop("pending_update", None)
    state.pop("installing", None)
    state.pop("last_failed", None)
    state["status"] = "current"
    _write_state(state_file, state)
    _append_log(log_file, f"updated to version {wanted_version}")
    return 0


def run_check(args: argparse.Namespace) -> int:
    if not args.loop:
        return check_once(args)

    rc = 0
    while True:
        try:
            rc = check_once(args)
        except Exception as exc:
            app_root = args.app_root.resolve()
            log_file = args.log_file or app_root / "update" / "auto_update.log"
            _append_log(log_file, f"ERROR: {exc}")
            rc = 1
        time.sleep(args.interval)
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental CAPI AI auto updater")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish", help="write latest.json for a patch ZIP")
    publish.add_argument("--package", type=Path, required=True, help="patch ZIP to publish")
    publish.add_argument("--output-dir", type=Path, required=True, help="directory served by HTTP/FTP")
    publish.add_argument("--manifest-name", default="latest.json")
    publish.add_argument("--version", default=None, help="override version read from ZIP")
    publish.add_argument("--package-url", default=None, help="absolute package URL; default is relative ZIP filename")
    publish.set_defaults(func=publish_manifest)

    check = subparsers.add_parser("check", help="check latest.json and stage a newer patch")
    check.add_argument("--manifest-url", required=True, help="http(s), ftp, file URL, or local path to latest.json")
    check.add_argument("--app-root", type=Path, default=PROJECT_ROOT)
    check.add_argument("--download-dir", type=Path, default=None)
    check.add_argument("--state-file", type=Path, default=None)
    check.add_argument("--log-file", type=Path, default=None)
    check.add_argument("--timeout", type=int, default=30)
    check.add_argument("--health-url", default=None, help="stored for the later manual apply")
    check.add_argument("--dry-run", action="store_true")
    check.add_argument("--loop", action="store_true", help="keep polling instead of checking once")
    check.add_argument("--interval", type=int, default=300, help="poll interval when --loop is used")
    check.set_defaults(func=run_check)

    apply_cmd = subparsers.add_parser("apply", help="manually install the staged update and restart")
    apply_cmd.add_argument("--app-root", type=Path, default=PROJECT_ROOT)
    apply_cmd.add_argument("--download-dir", type=Path, default=None)
    apply_cmd.add_argument("--state-file", type=Path, default=None)
    apply_cmd.add_argument("--log-file", type=Path, default=None)
    apply_cmd.add_argument("--install-command", default=None, help="defaults to <app-root>/install_patch.sh")
    apply_cmd.add_argument("--health-url", default=None, help="override the health URL stored by check")
    apply_cmd.add_argument("--no-auto-rollback", action="store_true")
    apply_cmd.add_argument("--delay", type=int, default=0, help="seconds to wait before install")
    apply_cmd.set_defaults(func=apply_pending)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
