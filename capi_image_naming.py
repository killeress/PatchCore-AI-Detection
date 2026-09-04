"""Shared AOI image filename prefix normalization."""
from pathlib import Path
from typing import Dict, Iterable, Tuple


CANONICAL_IMAGE_PREFIXES: Tuple[str, ...] = (
    "STANDARD",
    "WGF50500",
    "WGF00000",
    "G0F00000",
    "R0F00000",
    "W0F00000",
    "B0F00000",
)

IMAGE_PREFIX_ALIASES = {
    "U0F00000": "STANDARD",
}

# WHITEFRA is handled by the dedicated white-frame inspector, but its
# machine defect record is still part of the AOI report format.
AOI_REPORT_PREFIXES: Tuple[str, ...] = (
    tuple(IMAGE_PREFIX_ALIASES) + CANONICAL_IMAGE_PREFIXES + ("WHITEFRA",)
)


def canonical_image_prefix(image_name: str) -> str:
    stem = Path(str(image_name)).stem
    upper = stem.upper()

    for raw_prefix, canonical in IMAGE_PREFIX_ALIASES.items():
        if _matches_prefix(upper, raw_prefix):
            return canonical

    for prefix in CANONICAL_IMAGE_PREFIXES:
        if _matches_prefix(upper, prefix):
            return prefix

    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def source_image_prefix(image_name: str) -> str:
    stem = Path(str(image_name)).stem
    upper = stem.upper()
    for raw_prefix in IMAGE_PREFIX_ALIASES:
        if _matches_prefix(upper, raw_prefix):
            return raw_prefix
    return canonical_image_prefix(image_name)


def image_prefix_display_labels(image_names: Iterable[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for image_name in image_names:
        canonical = canonical_image_prefix(image_name)
        source = source_image_prefix(image_name)
        if source != canonical:
            labels[canonical] = source
        else:
            labels.setdefault(canonical, canonical)
    return labels


def panel_image_group_key(image_name: str) -> str:
    stem = Path(str(image_name)).stem
    upper = stem.upper()
    if upper.startswith("PINIGBI"):
        return "PINIGBI"
    if "OMIT0000" in upper:
        return "OMIT0000"
    return canonical_image_prefix(image_name)


def is_white_frame_image_name(image_name: str) -> bool:
    """Return whether a CAPI filename is a WHITEFRA image."""
    stem = Path(str(image_name)).stem.upper()
    return _matches_prefix(stem, "WHITEFRA")


def _matches_prefix(stem_upper: str, prefix: str) -> bool:
    if stem_upper == prefix or stem_upper.startswith(prefix + "_"):
        return True
    suffix = stem_upper[len(prefix):]
    return stem_upper.startswith(prefix) and len(suffix) == 6 and suffix.isdigit()
