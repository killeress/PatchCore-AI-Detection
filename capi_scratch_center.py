"""Central scratch collection and explicit, versioned model distribution."""
from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from capi_dataset_export import MANIFEST_FIELDS, SOURCE_MANIFEST_FIELDS, crop_patchcore_tile, write_manifest
from capi_station_adapter import create_station_adapter

TRANSFER_HEADER = "X-CAPI-Scratch-Transfer"
SAMPLE_LIMIT = 8 * 1024 * 1024
MODEL_LIMIT = 64 * 1024 * 1024
_LOCK = threading.RLock()
SAMPLE_MANIFEST_LOCK = threading.RLock()


def post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    parsed = urlsplit(url)
    address = ipaddress.ip_address(parsed.hostname or "")
    if parsed.scheme not in {"http", "https"} or address.version != 4 or address.packed[0] != 10:
        raise ValueError("中心／線體必須使用 10.x.x.x 的 HTTP(S) 位址")
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(str(address), parsed.port, timeout=timeout)
    try:
        connection.request("POST", parsed.path, json.dumps(payload).encode("utf-8"), {
            "Content-Type": "application/json", TRANSFER_HEADER: "1",
        })
        response = connection.getresponse()
        raw = response.read(1024 * 1024)
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise RuntimeError(f"中心／線體回應格式錯誤 (HTTP {response.status})")
        if response.status != 200 or not result.get("success"):
            raise RuntimeError(result.get("error") or f"HTTP {response.status}")
        return result
    finally:
        connection.close()


def store_sample(root: Path, payload: dict, source_ip: str) -> str:
    """One batch per source node; stable sample IDs make retries idempotent."""
    sample_id = str(payload.get("sample_id") or "")
    if not re.fullmatch(r"manual_[a-f0-9]{32}_[0-9]+", sample_id):
        raise ValueError("無效的刮痕樣本 ID")
    job = root / sample_id.rsplit("_", 1)[0]
    with SAMPLE_MANIFEST_LOCK:
        from capi_dataset_export import read_manifest
        manifest = job / "manifest.csv"
        rows = read_manifest(manifest) if manifest.exists() else {}
        previous = rows.get(sample_id)
        if previous and previous.get("source_ip") != source_ip:
            raise ValueError("此樣本屬於其他來源主機")
        if payload.get("action") == "remove":
            # Retain the crop for audit; non-ok rows are excluded by gallery/training.
            if manifest.exists():
                if sample_id in rows:
                    rows[sample_id]["status"] = "removed"
                    write_manifest(manifest, rows, fieldnames=list(rows[sample_id]))
            return sample_id
        if payload.get("action") != "save":
            raise ValueError("無效的刮痕樣本操作")
        raw = base64.b64decode(payload.get("png", ""), validate=True)
        if len(raw) > SAMPLE_LIMIT:
            raise ValueError("刮痕圖片過大")
        if hashlib.sha256(raw).hexdigest() != payload.get("sha256"):
            raise ValueError("刮痕圖片校驗失敗")
        if previous and previous.get("status") == "ok" and previous.get("crop_sha256") == payload["sha256"]:
            return sample_id  # A network retry must not overwrite a central reviewer's relabel.
        # Require the 512px PNG produced by the source; bound dimensions before decode.
        if (raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR"
                or int.from_bytes(raw[16:20], "big") != 512
                or int.from_bytes(raw[20:24], "big") != 512):
            raise ValueError("刮痕樣本必須是 512×512 PNG")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("無法解碼刮痕圖片")
        metadata = payload.get("metadata") or {}
        row = {key: str(metadata.get(key, ""))[:1024] for key in MANIFEST_FIELDS + SOURCE_MANIFEST_FIELDS}
        adapter = create_station_adapter(row.get("station_profile") or "capi")
        prefix = adapter.training_image_prefix(row["image_name"])
        if prefix not in adapter.training_prefixes:
            raise ValueError("此光源不支援刮痕樣本")
        crop_rel = f"over_surface_scratch/{prefix}/crop/{sample_id}.png"
        row.update(sample_id=sample_id, label="over_surface_scratch", prefix=prefix,
                   source_type="patchcore_tile", crop_path=crop_rel, heatmap_path="",
                   status="ok", collected_at=datetime.now().isoformat(),
                   over_review_category="surface_scratch", source_ip=source_ip,
                   sample_source="inference_record", crop_sha256=payload["sha256"],
                   station_profile=adapter.profile)
        crop_path = job / crop_rel
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = crop_path.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.replace(crop_path)
        rows[sample_id] = row
        fields = list(dict.fromkeys(key for item in rows.values() for key in item))
        write_manifest(manifest, rows, fieldnames=fields)
    return sample_id


class ScratchCenterMixin:
    def _scratch_center_ip(self):
        return self._load_central_account_location()["ip"]

    def _scratch_is_center(self):
        # Use the actual socket address, never a client-controlled Host header.
        try:
            return self.connection.getsockname()[0] == self._scratch_center_ip()
        except (AttributeError, OSError):
            return False

    def _scratch_lines(self):
        config = self.db.get_central_dashboard_config(self._load_central_dashboard_file_config())
        center_prefix = self._scratch_center_ip().split(".")[:2]
        lines = []
        for line in config.get("lines", []):
            parsed = urlsplit(str(line.get("apiUrl") or ""))
            if line.get("enabled") is not False and (parsed.hostname or "").split(".")[:2] == center_prefix:
                lines.append(line)
        return lines

    def _handle_scratch_center_entry(self):
        self._redirect("http://" + self._scratch_center_ip() + "/scratch-management")

    def _handle_scratch_management(self):
        if not self._scratch_is_center():
            self._handle_scratch_center_entry()
            return
        bundles = sorted(p.name for p in (Path(__file__).parent / "deployment").glob("scratch_classifier*.pkl"))
        self._send_response(200, self.jinja_env.get_template("scratch_management.html").render(
            request_path="/scratch-management", bundles=bundles, lines=self._scratch_lines(),
            dataset_root=str(self._dataset_export_base_dir()),
        ))

    def _scratch_transfer_body(self, limit):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > limit:
                raise ValueError("傳輸大小超出限制")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("傳輸內容必須是 JSON object")
            return data
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return None

    def _handle_scratch_sample_receive(self):
        peer = self.client_address[0]
        allowed = {urlsplit(line["apiUrl"]).hostname for line in self._scratch_lines()}
        allowed.add(self._scratch_center_ip())
        if (not self._scratch_is_center() or peer not in allowed
                or self.headers.get(TRANSFER_HEADER) != "1"):
            self._send_json({"error": "來源不是中心已啟用的線體"}, status=403)
            return
        data = self._scratch_transfer_body(SAMPLE_LIMIT * 2)
        if data is None:
            return
        try:
            sample_id = store_sample(self._dataset_export_base_dir(), data, peer)
            self._send_json({"success": True, "sample_id": sample_id})
        except (ValueError, OSError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _sync_record_scratch(self, record, candidate, existing=None, remove=False):
        tile_id = int(candidate["tile_result_id"])
        center = existing["center_ip"] if existing else self._scratch_center_ip()
        if existing:
            sample_id = existing["sample_id"]
        else:
            with _LOCK:
                node = self.db.get_config_param("scratch_source_node_id")
                node_id = node["decoded_value"] if node else uuid.uuid4().hex
                if not node:
                    self.db.update_config_param("scratch_source_node_id", node_id, "刮痕樣本來源識別")
            sample_id = f"manual_{node_id}_{tile_id}"
        payload = {"sample_id": sample_id, "action": "remove" if remove else "save"}
        if not remove:
            if candidate.get("is_bomb") or candidate.get("image_is_bomb") or candidate.get("is_exclude_zone"):
                raise ValueError("爆點或排除區 Tile 不能加入刮痕資料集")
            source = self._mes_review_resolve_source(candidate.get("image_path") or "")
            image = self._read_inference_image(source, cv2.IMREAD_UNCHANGED) if source.is_file() else None
            if image is None:
                raise ValueError("原圖不存在或無法讀取")
            crop = crop_patchcore_tile(image, int(candidate.get("tile_x") or 0),
                                      int(candidate.get("tile_y") or 0),
                                      int(candidate.get("tile_w") or 512), int(candidate.get("tile_h") or 512))
            ok, png = cv2.imencode(".png", crop)
            if not ok:
                raise ValueError("刮痕裁圖編碼失敗")
            raw = png.tobytes()
            payload.update(png=base64.b64encode(raw).decode("ascii"), sha256=hashlib.sha256(raw).hexdigest(), metadata={
                "glass_id": record.get("glass_id", ""), "image_name": candidate.get("image_name", ""),
                "inference_record_id": record["id"], "image_result_id": candidate["image_result_id"],
                "tile_idx": candidate.get("tile_id", tile_id), "ai_score": candidate.get("ai_score", 0),
                "defect_x": candidate.get("aoi_image_x", ""), "defect_y": candidate.get("aoi_image_y", ""),
                "inference_timestamp": record.get("request_time", ""),
                "machine_id": record.get("model_id", ""), "machine_no": record.get("machine_no", ""),
                "station_profile": self._station_adapter().profile,
                "over_review_note": "推論紀錄人工歸類",
            })
        if self._scratch_is_center() and center == self._scratch_center_ip():
            store_sample(self._dataset_export_base_dir(), payload, center)
        else:
            result = post_json(f"http://{center}/api/scratch/samples", payload)
            if result.get("sample_id") != sample_id:
                raise RuntimeError("中心回覆的樣本 ID 不符，請重試")
        self.db.set_scratch_sample_classification(tile_id, "" if remove else sample_id, center)
        return sample_id

    def _handle_scratch_distribute(self):
        if not self._scratch_is_center():
            self._send_json({"error": "請由中心下發"}, status=403)
            return
        user = self._require_settings_user(api=True, admin=True)
        if not user:
            return
        data = self._read_json_body()
        if not isinstance(data, dict):
            return
        name = str(data.get("bundle") or "")
        if not re.fullmatch(r"scratch_classifier[A-Za-z0-9_.-]*\.pkl", name):
            self._send_json({"error": "無效模型名稱"}, status=400)
            return
        path = Path(__file__).parent / "deployment" / name
        if not path.is_file() or path.stat().st_size > MODEL_LIMIT:
            self._send_json({"error": "模型不存在或過大"}, status=400)
            return
        line = next((line for line in self._scratch_lines() if line["id"] == data.get("line_id")), None)
        if not line:
            self._send_json({"error": "請選擇已啟用的線體"}, status=400)
            return
        raw = path.read_bytes()
        parsed = urlsplit(line["apiUrl"])
        try:
            result = post_json(f"{parsed.scheme}://{parsed.netloc}/api/scratch/model", {
                "name": name, "sha256": hashlib.sha256(raw).hexdigest(),
                "bundle": base64.b64encode(raw).decode("ascii"),
                "requested_by": user.get("username", ""),
            }, timeout=120)
            self._send_json(result)
        except (ValueError, OSError, RuntimeError, http.client.HTTPException) as exc:
            self._send_json({"error": f"下發失敗：{exc}"}, status=502)

    def _handle_scratch_model_receive(self):
        if self.client_address[0] != self._scratch_center_ip() or self.headers.get(TRANSFER_HEADER) != "1":
            self._send_json({"error": "只接受已設定中心的模型"}, status=403)
            return
        data = self._scratch_transfer_body(MODEL_LIMIT * 2)
        if data is None:
            return
        try:
            result = self._install_scratch_model(data)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": f"模型未啟用：{exc}"}, status=400)

    def _install_scratch_model(self, data):
        from scratch_classifier import ScratchClassifier, load_bundle
        raw = base64.b64decode(data.get("bundle", ""), validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) > MODEL_LIMIT or digest != data.get("sha256"):
            raise ValueError("模型校驗失敗")
        name = str(data.get("name") or "")
        if not re.fullmatch(r"scratch_classifier[A-Za-z0-9_.-]*\.pkl", name):
            raise ValueError("無效模型名稱")
        # Content-addressed destination avoids collisions with a line's local vN.
        relative = Path("deployment") / f"{Path(name).stem}_{digest[:16]}.pkl"
        destination = Path(__file__).parent / relative
        server = self._capi_server_instance
        config = server.config
        with _LOCK:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".pending")
            try:
                temporary.write_bytes(raw)
                repo = config.scratch_dinov2_repo_path
                weights = config.scratch_dinov2_weights_path
                if not repo or not Path(repo).is_dir() or not weights or not Path(weights).is_file():
                    raise ValueError("線體缺少本地 DINOv2 repo／權重，請先完成離線部署")
                # Validate structure, local base weights and a real forward pass before activation.
                classifier = ScratchClassifier(temporary, weights, repo, device="cpu")
                lora_weights, _, metadata, _ = load_bundle(temporary)
                expected = {key for key in classifier._model.state_dict()
                            if "lora_A" in key or "lora_B" in key}
                if set(lora_weights) != expected or not expected:
                    raise ValueError("LoRA 權重不完整")
                if not np.isfinite(metadata.conformal_threshold) or not 0 <= metadata.conformal_threshold <= 1:
                    raise ValueError("模型校準門檻無效")
                score = classifier.predict(np.zeros((512, 512, 3), dtype=np.uint8))
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("模型試跑分數無效")
                del classifier
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            previous = config.scratch_bundle_path
            self.db.activate_scratch_bundle(relative.as_posix(),
                                            f"中心下發 SHA256={digest}; previous={previous}",
                                            changed_by=str(data.get("requested_by") or "central"))
            configs = [config]
            inferencers = list((getattr(server, "inferencers", None) or {}).values())
            if self.inferencer is not None:
                inferencers.append(self.inferencer)
            configs.extend(inferencer.config for inferencer in inferencers)
            for target in configs:
                target.scratch_bundle_path = relative.as_posix()
                target.scratch_classifier_enabled = True
        return {"success": True, "bundle": relative.as_posix(), "sha256": digest,
                "previous_bundle": previous, "message": "模型已驗證並切換，下次推論載入新版本"}
