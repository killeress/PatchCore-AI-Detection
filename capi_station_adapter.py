"""Station-specific AOI input conventions.

The inference core uses stable internal lighting keys.  The deployment station
is selected from the Linux hostname; filenames and report paths are never used
to guess the station.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Dict, List, Mapping, Optional, Tuple

from capi_image_naming import (
    canonical_image_prefix,
    panel_image_group_key,
    source_image_prefix,
)


@dataclass(frozen=True)
class StationAOIDefect:
    defect_code: str
    x: int
    y: int
    image_prefix: str
    coordinate_space: str = "product"


class StationAdapter:
    profile = "capi"
    inference_prefixes: Tuple[str, ...] = (
        "G0F00000",
        "R0F00000",
        "W0F00000",
        "WGF50500",
        "STANDARD",
    )
    boundary_reference_priority: Tuple[str, ...] = (
        "W0F00000",
        "STANDARD",
        "G0F00000",
        "R0F00000",
        "WGF50500",
    )

    def image_prefix(self, image_name: str) -> str:
        return canonical_image_prefix(image_name)

    def model_prefix(self, image_prefix: str) -> str:
        return image_prefix

    def report_prefix(self, image_name: str) -> str:
        return source_image_prefix(image_name)

    def image_group_key(self, image_name: str) -> str:
        return panel_image_group_key(image_name)

    def is_omit_image(self, image_name: str) -> bool:
        stem = Path(image_name).stem.upper()
        return stem.startswith("PINIGBI") or "OMIT0000" in stem

    def is_white_frame_image(self, image_name: str) -> bool:
        return Path(image_name).stem.upper().startswith("WHITEFRA_")

    def find_white_frame_image(self, panel_dir: Path) -> Optional[Path]:
        return _latest_matching_image(panel_dir, self.is_white_frame_image)

    def is_inference_prefix(self, image_prefix: str) -> bool:
        return image_prefix != "WHITEFRA"

    def parse_aoi_report(
        self,
        panel_dir: Path,
        *,
        glass_id: str,
        machine_judgment: str,
    ) -> Optional[Dict[str, List[StationAOIDefect]]]:
        # None tells CAPIInferencer to retain the existing per-panel TXT parser.
        return None


class AAPIStationAdapter(StationAdapter):
    profile = "aapi"
    inference_prefixes = (
        "R0F00000",
        "W0F00000",
        "W0F00010",
        "WGF50500",
        "WINDOWS_BG",
    )
    boundary_reference_priority = (
        "W0F00000",
        "WINDOWS_BG",
        "R0F00000",
        "W0F00010",
        "WGF50500",
    )

    _SOURCE_PREFIXES: Tuple[Tuple[str, str], ...] = (
        ("WHITE_FRAME", "WHITEFRA"),
        ("WINDOWS_BG", "WINDOWS_BG"),
        ("W0F00010", "W0F00010"),
        ("WGF50500", "WGF50500"),
        ("R0F00000", "R0F00000"),
        ("W0F00000", "W0F00000"),
        ("B0F00000", "B0F00000"),
        ("PINIGBI0", "PINIGBI"),
    )
    _MODEL_ALIASES = {
        "WINDOWS_BG": "STANDARD",
        "W0F00010": "WGF50500",
    }
    _REPORT_RECORD = re.compile(
        r"(White_Frame|Windows_BG|W0F00010|WGF50500|R0F00000|W0F00000|B0F00000),"
        r"([A-Za-z0-9]+)\((\d+),(\d+)\)",
        re.IGNORECASE,
    )
    _DATE_SEGMENT = re.compile(r"(?<!\d)(20\d{6})(?!\d)")

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        config = config or {}
        self.report_root = Path(str(config.get("report_root") or "/192.168.2.190/LOG"))
        self.report_retry_count = max(1, int(config.get("report_retry_count") or 3))
        self.report_retry_interval_seconds = max(
            0.0,
            float(config.get("report_retry_interval_seconds") or 0.2),
        )

    def image_prefix(self, image_name: str) -> str:
        stem = Path(str(image_name)).stem
        upper = stem.upper()
        for source, internal in self._SOURCE_PREFIXES:
            marker = source.upper()
            index = upper.rfind(marker)
            if index <= 0:
                continue
            suffix = upper[index + len(marker):]
            if len(suffix) == 6 and suffix.isdigit():
                return internal
        return stem

    def model_prefix(self, image_prefix: str) -> str:
        return self._MODEL_ALIASES.get(str(image_prefix).upper(), image_prefix)

    def report_prefix(self, image_name: str) -> str:
        return self.model_prefix(self.image_prefix(image_name))

    def image_group_key(self, image_name: str) -> str:
        return self.image_prefix(image_name)

    def is_omit_image(self, image_name: str) -> bool:
        return self.image_prefix(image_name) == "PINIGBI"

    def is_white_frame_image(self, image_name: str) -> bool:
        return self.image_prefix(image_name) == "WHITEFRA"

    def parse_aoi_report(
        self,
        panel_dir: Path,
        *,
        glass_id: str,
        machine_judgment: str,
    ) -> Optional[Dict[str, List[StationAOIDefect]]]:
        glass_id = str(glass_id or "").strip()
        if not glass_id:
            raise RuntimeError("AAPI AOI report requires glass_id")
        if str(machine_judgment or "").strip().upper() == "OK":
            return {}

        report_file = self._report_file_for_panel(panel_dir)
        last_error = ""
        for attempt in range(self.report_retry_count):
            try:
                parsed, last_error = self._read_latest_glass_record(report_file, glass_id)
            except OSError as exc:
                parsed = None
                last_error = f"{type(exc).__name__}: {exc}"
            if parsed is not None:
                if not parsed:
                    last_error = "latest_record_status_ok"
                else:
                    return parsed
            if attempt + 1 < self.report_retry_count:
                time.sleep(self.report_retry_interval_seconds)

        raise RuntimeError(
            f"AAPI AOI report unavailable for glass={glass_id}: "
            f"file={report_file} reason={last_error or 'not_found'}"
        )

    def _report_file_for_panel(self, panel_dir: Path) -> Path:
        match = self._DATE_SEGMENT.search(str(panel_dir))
        if match is None:
            raise RuntimeError(f"AAPI image path has no YYYYMMDD date segment: {panel_dir}")
        yyyymmdd = match.group(1)
        report_date = yyyymmdd[2:4] + yyyymmdd[4:6] + yyyymmdd[6:8]
        return self.report_root / f"Report{report_date}.log"

    def _read_latest_glass_record(
        self,
        report_file: Path,
        glass_id: str,
    ) -> Tuple[Optional[Dict[str, List[StationAOIDefect]]], str]:
        text = report_file.read_text(encoding="utf-8", errors="replace")
        matching_lines = []
        for line in text.splitlines():
            fields = line.strip().split(",", 3)
            if len(fields) >= 3 and fields[1].strip() == glass_id:
                matching_lines.append(fields)
        if not matching_lines:
            return None, "not_found"

        fields = matching_lines[-1]
        status = fields[2].strip().upper()
        payload = fields[3].strip() if len(fields) == 4 else ""
        if status == "OK" and not payload:
            return {}, ""
        if status != "NG" or not payload:
            return None, "latest_record_incomplete"

        matches = list(self._REPORT_RECORD.finditer(payload))
        residue = self._REPORT_RECORD.sub("", payload).strip(" ,;\t")
        if not matches or residue:
            return None, "latest_record_incomplete"

        parsed: Dict[str, List[StationAOIDefect]] = {}
        for match in matches:
            source_prefix, defect_code, raw_x, raw_y = match.groups()
            internal_prefix = self._internal_report_prefix(source_prefix)
            parsed.setdefault(internal_prefix, []).append(StationAOIDefect(
                defect_code=defect_code,
                x=int(raw_x),
                y=int(raw_y),
                image_prefix=internal_prefix,
                coordinate_space="image",
            ))
        return parsed, ""

    def _internal_report_prefix(self, source_prefix: str) -> str:
        upper = str(source_prefix).upper()
        for source, internal in self._SOURCE_PREFIXES:
            if upper == source:
                return internal
        return str(source_prefix)


def resolve_station_profile_from_hostname(
    hostname: str,
    *,
    default_if_unknown: Optional[str] = None,
) -> str:
    """Resolve CAPI/AAPI from one hostname without inspecting request data."""
    normalized = str(hostname or "").strip().casefold()
    has_aapi = "aapi" in normalized
    has_capi = "capi" in normalized

    if has_aapi and not has_capi:
        return "aapi"
    if has_capi and not has_aapi:
        return "capi"
    if has_aapi and has_capi:
        raise RuntimeError(
            f"Ambiguous station hostname {hostname!r}: contains both 'aapi' and 'capi'"
        )

    if default_if_unknown is not None:
        fallback = str(default_if_unknown).strip().casefold()
        if fallback not in {"capi", "aapi"}:
            raise ValueError(
                "default_if_unknown must be either 'capi' or 'aapi'"
            )
        return fallback

    raise RuntimeError(
        f"Unknown station hostname {hostname!r}: expected hostname to contain "
        "either 'aapi' or 'capi'"
    )


def create_station_adapter(
    profile: str,
    config: Optional[Mapping[str, object]] = None,
) -> StationAdapter:
    normalized = str(profile or "capi").strip().lower()
    if normalized == "capi":
        return StationAdapter()
    if normalized == "aapi":
        return AAPIStationAdapter(config)
    raise ValueError(f"Unsupported station_profile: {profile!r}; expected 'capi' or 'aapi'")


def _latest_matching_image(panel_dir: Path, predicate) -> Optional[Path]:
    candidates = [
        path
        for path in Path(panel_dir).iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        and predicate(path.name)
    ]
    if not candidates:
        return None

    def latest_key(path: Path) -> Tuple[int, str]:
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = 0
        return modified, path.name

    return max(candidates, key=latest_key)
