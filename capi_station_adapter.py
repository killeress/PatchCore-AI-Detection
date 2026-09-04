"""Station-specific AOI input conventions.

The inference core uses stable internal lighting keys.  The deployment station
is selected from the Linux hostname; filenames and report paths are never used
to guess the station.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

from capi_image_naming import (
    canonical_image_prefix,
    is_white_frame_image_name,
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

    def training_image_prefix(self, image_name: str) -> str:
        """Return the PatchCore model lighting used by a source image."""
        return self.model_prefix(self.image_prefix(image_name))

    @property
    def training_prefixes(self) -> Tuple[str, ...]:
        """Model lightings that can be trained from this station's images."""
        available = []
        for prefix in self.inference_prefixes:
            if not self.is_inference_prefix(prefix):
                continue
            model_prefix = self.model_prefix(prefix)
            if model_prefix not in available:
                available.append(model_prefix)
        return tuple(available)

    def report_prefix(self, image_name: str) -> str:
        return source_image_prefix(image_name)

    def image_group_key(self, image_name: str) -> str:
        return panel_image_group_key(image_name)

    def is_omit_image(self, image_name: str) -> bool:
        stem = Path(image_name).stem.upper()
        return stem.startswith("PINIGBI") or "OMIT0000" in stem

    def is_white_frame_image(self, image_name: str) -> bool:
        return is_white_frame_image_name(image_name)

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
        report_payload: str = "",
    ) -> Optional[Dict[str, List[StationAOIDefect]]]:
        # None tells CAPIInferencer to retain the existing per-panel TXT parser.
        return None


class AAPIStationAdapter(StationAdapter):
    profile = "aapi"
    inference_prefixes = (
        "G0F00000",
        "R0F00000",
        "W0F00000",
        "WGF25250",
        "W0F00010",
        "WGF50500",
        "U0F00000",
        "WINDOWS_BG",
    )
    boundary_reference_priority = (
        "W0F00000",
        "WINDOWS_BG",
        "U0F00000",
        "G0F00000",
        "R0F00000",
        "WGF25250",
        "W0F00010",
        "WGF50500",
    )

    _SOURCE_PREFIXES: Tuple[Tuple[str, str], ...] = (
        ("WHITE_FRAME", "WHITEFRA"),
        ("BWFRAME0", "WHITEFRA"),
        ("WINDOWS_BG", "WINDOWS_BG"),
        ("STANDARD", "WINDOWS_BG"),
        ("W0F00010", "W0F00010"),
        ("WGF25250", "WGF25250"),
        ("WGF50500", "WGF50500"),
        ("G0F00000", "G0F00000"),
        ("R0F00000", "R0F00000"),
        ("W0F00000", "W0F00000"),
        ("U0F00000", "U0F00000"),
        ("B0F00000", "B0F00000"),
        ("PINIGBI0", "PINIGBI"),
    )
    _MODEL_ALIASES = {
        "WINDOWS_BG": "STANDARD",
    }
    _REPORT_RECORD = re.compile(
        r"(White_Frame|BWFRAME0|Windows_BG|STANDARD|W0F00010|WGF25250|"
        r"WGF50500|G0F00000|R0F00000|W0F00000|U0F00000|B0F00000),"
        r"([A-Za-z0-9]+)\((\d+),(\d+)\)",
        re.IGNORECASE,
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
        report_payload: str = "",
    ) -> Optional[Dict[str, List[StationAOIDefect]]]:
        glass_id = str(glass_id or "").strip()
        if not glass_id:
            raise RuntimeError("AAPI AOI coordinates require glass_id")

        judgment = str(machine_judgment or "").strip().upper()
        if judgment != "NG":
            return {}

        payload = str(report_payload or "").strip()
        if not payload:
            raise RuntimeError(
                f"AAPI AOI coordinates missing from Testing request for glass={glass_id}"
            )

        matches = list(self._REPORT_RECORD.finditer(payload))
        residue = self._REPORT_RECORD.sub("", payload).strip(" ,;\t")
        if not matches or residue:
            raise RuntimeError(
                f"AAPI AOI coordinates malformed for glass={glass_id}: {payload[:100]}"
            )

        parsed: Dict[str, List[StationAOIDefect]] = {}
        for match in matches:
            source_prefix, defect_code, raw_x, raw_y = match.groups()
            internal_prefix = self._internal_report_prefix(source_prefix)
            parsed.setdefault(internal_prefix, []).append(StationAOIDefect(
                defect_code=defect_code,
                # AAPI records product X in three-channel units.  Convert it
                # before mapping the product coordinate with protocol resolution.
                x=int(raw_x) // 3,
                y=int(raw_y),
                image_prefix=internal_prefix,
                coordinate_space="product",
            ))
        return parsed

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


def create_station_adapter(profile: str) -> StationAdapter:
    normalized = str(profile or "capi").strip().lower()
    if normalized == "capi":
        return StationAdapter()
    if normalized == "aapi":
        return AAPIStationAdapter()
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
