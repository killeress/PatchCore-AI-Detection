"""Dot-matrix panel mark detection for debug inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


_GRID_ROWS = 7
_GRID_COLS = 5

_PATTERNS: Dict[str, List[Tuple[str, ...]]] = {
    "0": [
        ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
        ("11111", "11011", "11011", "11011", "11011", "11011", "11111"),
    ],
    "1": [("00100", "01100", "00100", "00100", "00100", "00100", "01110")],
    "2": [("01110", "10001", "00001", "00010", "00100", "01000", "11111")],
    "3": [("11110", "00001", "00001", "01110", "00001", "00001", "11110")],
    "4": [("00010", "00110", "01010", "10010", "11111", "00010", "00010")],
    "5": [
        ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
        ("11111", "11000", "11110", "11111", "00011", "11011", "11111"),
    ],
    "6": [("01110", "10000", "10000", "11110", "10001", "10001", "01110")],
    "7": [("11111", "00001", "00010", "00100", "01000", "01000", "01000")],
    "8": [("01110", "10001", "10001", "01110", "10001", "10001", "01110")],
    "9": [("01110", "10001", "10001", "01111", "00001", "00001", "01110")],
    "A": [
        ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        ("00100", "01110", "10001", "11111", "10001", "10001", "10001"),
    ],
    "B": [("11110", "10001", "10001", "11110", "10001", "10001", "11110")],
    "C": [("01111", "10000", "10000", "10000", "10000", "10000", "01111")],
    "D": [("11110", "10001", "10001", "10001", "10001", "10001", "11110")],
    "E": [
        ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        ("11111", "11000", "11111", "11000", "11111", "00000", "00000"),
    ],
    "F": [("11111", "10000", "10000", "11110", "10000", "10000", "10000")],
    "G": [("01111", "10000", "10000", "10011", "10001", "10001", "01111")],
    "H": [("10001", "10001", "10001", "11111", "10001", "10001", "10001")],
    "I": [("11111", "00100", "00100", "00100", "00100", "00100", "11111")],
    "J": [
        ("11111", "00001", "00001", "00001", "00001", "10001", "01110"),
        ("01111", "01111", "00110", "00110", "00110", "11110", "11100"),
    ],
    "K": [("10001", "10010", "10100", "11000", "10100", "10010", "10001")],
    "L": [("10000", "10000", "10000", "10000", "10000", "10000", "11111")],
    "M": [("10001", "11011", "10101", "10101", "10001", "10001", "10001")],
    "N": [("10001", "11001", "10101", "10011", "10001", "10001", "10001")],
    "O": [("01110", "10001", "10001", "10001", "10001", "10001", "01110")],
    "P": [("11110", "10001", "10001", "11110", "10000", "10000", "10000")],
    "Q": [("01110", "10001", "10001", "10001", "10101", "10010", "01101")],
    "R": [("11110", "10001", "10001", "11110", "10100", "10010", "10001")],
    "S": [("01111", "10000", "10000", "01110", "00001", "00001", "11110")],
    "T": [("11111", "00100", "00100", "00100", "00100", "00100", "00100")],
    "U": [("10001", "10001", "10001", "10001", "10001", "10001", "01110")],
    "V": [("10001", "10001", "10001", "10001", "10001", "01010", "00100")],
    "W": [("10001", "10001", "10001", "10101", "10101", "10101", "01010")],
    "X": [("10001", "10001", "01010", "00100", "01010", "10001", "10001")],
    "Y": [("10001", "10001", "01010", "00100", "00100", "00100", "00100")],
    "Z": [("11111", "00001", "00010", "00100", "01000", "10000", "11111")],
}

_POSITIONAL_PATTERNS: Dict[Tuple[str, int], List[Tuple[str, ...]]] = {
    ("B", 0): [("01110", "11111", "11010", "11110", "11011", "11111", "11110")],
    # 實機網紋會讓點陣跨格；保留 R 的中段與下斜腳，避免被 F/P 吸走。
    ("R", 0): [("11110", "11011", "10001", "11110", "10100", "10010", "10000")],
    # 第二碼 5 的實機跨格容錯，避免低對比畫面被判成 J/E。
    ("5", 1): [("01111", "10000", "01000", "11110", "00001", "00011", "11110")],
}

_FIRST_CHARS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_SECOND_CHARS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

_ROI_RATIOS = (
    ("top_right", (0.72, 0.03, 0.99, 0.45)),
    ("bottom_left", (0.01, 0.52, 0.32, 0.97)),
)


def detect_panel_mark_from_path(image_path: str | Path, include_debug: bool = False) -> Dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"found": False, "error": f"cannot read image: {image_path}"}
    return detect_panel_mark(image, include_debug=include_debug)


def detect_panel_mark(image: np.ndarray, include_debug: bool = False) -> Dict[str, Any]:
    gray = _to_gray8(image)
    height, width = gray.shape[:2]

    candidates: List[Dict[str, Any]] = []
    for roi_name, ratios in _ROI_RATIOS:
        x1, y1, x2, y2 = _roi_from_ratios(width, height, ratios)
        roi = gray[y1:y2, x1:x2]
        candidate = _detect_roi(roi, x1, y1, width, height, roi_name)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return {
            "found": False,
            "message": "no dot-matrix mark candidate found",
            "candidates": [],
        }

    best = max(candidates, key=lambda item: item["candidate_score"])
    public = _public_result(best, candidates)

    if include_debug:
        public["_debug_images"] = _make_debug_images(gray, best)

    return public


def _to_gray8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint8:
        return image

    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _roi_from_ratios(width: int, height: int, ratios: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(width * ratios[0])))
    y1 = max(0, min(height - 1, int(height * ratios[1])))
    x2 = max(x1 + 1, min(width, int(width * ratios[2])))
    y2 = max(y1 + 1, min(height, int(height * ratios[3])))
    return x1, y1, x2, y2


def _detect_roi(
    roi: np.ndarray,
    offset_x: int,
    offset_y: int,
    full_width: int,
    full_height: int,
    roi_name: str,
) -> Optional[Dict[str, Any]]:
    filtered = _dot_mask(roi, min(full_width, full_height))
    groups = _find_mark_groups(filtered, offset_x, offset_y, full_width, full_height)
    if not groups:
        return None

    best: Optional[Dict[str, Any]] = None
    for group in groups:
        for recognized in _recognize_group_variants(group["mask"]):
            char_scores = [ch["score"] for ch in recognized["chars"]]
            mean_score = float(np.mean(char_scores)) if char_scores else 0.0
            candidate_score = (
                mean_score
                + min(group["component_count"] / 40.0, 1.0) * 0.08
                + _orientation_prior(roi_name, recognized["orientation"])
            )
            candidate = {
                **group,
                **recognized,
                "roi": roi_name,
                "candidate_score": candidate_score,
                "confidence": float(np.clip((mean_score - 0.05) / 0.25, 0.0, 1.0)),
            }
            if best is None or candidate["candidate_score"] > best["candidate_score"]:
                best = candidate

    return best


def _dot_mask(roi: np.ndarray, scale_basis: int) -> np.ndarray:
    sigma = max(17, int(round(scale_basis * 0.007)))
    background = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
    enhanced = cv2.subtract(background, roi)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    filtered = np.zeros_like(binary)
    max_blob = max(24, int(round(scale_basis * 0.014)))
    for idx in range(1, count):
        x, y, width, height, area = stats[idx]
        if 3 <= width <= max_blob and 3 <= height <= max_blob and 5 <= area <= max_blob * max_blob:
            filtered[y : y + height, x : x + width] = binary[y : y + height, x : x + width]

    return filtered


def _remove_tiny_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = [
        int(stats[idx, cv2.CC_STAT_AREA])
        for idx in range(1, count)
        if stats[idx, cv2.CC_STAT_WIDTH] >= 2
        and stats[idx, cv2.CC_STAT_HEIGHT] >= 2
        and stats[idx, cv2.CC_STAT_AREA] >= 5
    ]
    if not areas:
        return mask

    min_area = max(5, int(round(float(np.median(areas)) * 0.15)))
    keep = np.zeros(count, dtype=np.uint8)
    for idx in range(1, count):
        if (
            stats[idx, cv2.CC_STAT_WIDTH] >= 2
            and stats[idx, cv2.CC_STAT_HEIGHT] >= 2
            and stats[idx, cv2.CC_STAT_AREA] >= min_area
        ):
            keep[idx] = 255
    return keep[labels]


def _find_mark_groups(
    mask: np.ndarray,
    offset_x: int,
    offset_y: int,
    full_width: int,
    full_height: int,
) -> List[Dict[str, Any]]:
    join = max(15, int(round(min(full_width, full_height) * 0.010)))
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (join, join)), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, 8)

    groups: List[Dict[str, Any]] = []
    for idx in range(1, count):
        x, y, width, height, area = stats[idx]
        if area < 1000:
            continue

        component_mask = np.where(labels == idx, 255, 0).astype(np.uint8)
        mark_mask = cv2.bitwise_and(mask, mask, mask=component_mask)
        mark_mask = _remove_tiny_components(mark_mask)
        ys, xs = np.where(mark_mask > 0)
        if len(xs) == 0:
            continue

        tight_x1, tight_x2 = int(xs.min()), int(xs.max()) + 1
        tight_y1, tight_y2 = int(ys.min()), int(ys.max()) + 1
        mark_width = tight_x2 - tight_x1
        mark_height = tight_y2 - tight_y1
        if not _plausible_mark_size(mark_width, mark_height, full_width, full_height):
            continue

        tight = mark_mask[tight_y1:tight_y2, tight_x1:tight_x2]
        comp_count = _count_small_components(tight)
        if comp_count < 10:
            continue

        groups.append(
            {
                "bbox": {
                    "x": offset_x + tight_x1,
                    "y": offset_y + tight_y1,
                    "width": mark_width,
                    "height": mark_height,
                },
                "mask": tight,
                "component_count": comp_count,
            }
        )

    return groups


def _plausible_mark_size(mark_width: int, mark_height: int, full_width: int, full_height: int) -> bool:
    if mark_width <= 0 or mark_height <= 0:
        return False

    min_w = max(30, int(full_width * 0.004))
    max_w = max(120, int(full_width * 0.080))
    min_h = max(30, int(full_height * 0.006))
    max_h = max(120, int(full_height * 0.100))
    ratio = mark_width / float(mark_height)

    return min_w <= mark_width <= max_w and min_h <= mark_height <= max_h and 0.70 <= ratio <= 2.40


def _orientation_prior(roi_name: str, orientation: str) -> float:
    if roi_name == "top_right" and orientation == "normal":
        return 0.03
    if roi_name == "bottom_left" and orientation == "rot180":
        return 0.03
    return 0.0


def _count_small_components(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    small = 0
    for idx in range(1, count):
        _, _, width, height, area = stats[idx]
        if area >= 5 and width >= 2 and height >= 2:
            small += 1
    return small


def _recognize_group(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    split = _split_two_chars(mask)
    if split is None:
        return None

    row_correction = _estimate_dot_row_correction([char_mask for char_mask, _ in split])
    chars = []
    text_parts = []
    char_boxes = []
    for idx, (char_mask, box) in enumerate(split):
        if idx == 0 and row_correction != 0.0:
            char_mask = _apply_dot_row_correction(char_mask, row_correction)
        allowed = _FIRST_CHARS if idx == 0 else _SECOND_CHARS
        recognized = _recognize_char(char_mask, allowed, char_index=idx)
        if recognized is None:
            return None
        chars.append(recognized)
        text_parts.append(recognized["char"])
        char_boxes.append(box)

    return {
        "text": "".join(text_parts),
        "chars": chars,
        "char_boxes": char_boxes,
    }


def _recognize_group_variants(mask: np.ndarray) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for orientation, oriented_mask in (
        ("normal", mask),
        ("rot180", cv2.rotate(mask, cv2.ROTATE_180)),
    ):
        recognized = _recognize_group(oriented_mask)
        if recognized is None:
            continue

        if orientation == "rot180":
            recognized["char_boxes"] = [
                _rotate_box_180(box, mask.shape[1], mask.shape[0])
                for box in recognized["char_boxes"]
            ]

        recognized["orientation"] = orientation
        recognized["mask"] = oriented_mask
        variants.append(recognized)
    return variants


def _rotate_box_180(box: Dict[str, int], width: int, height: int) -> Dict[str, int]:
    return {
        "x": int(width - box["x"] - box["width"]),
        "y": int(height - box["y"] - box["height"]),
        "width": int(box["width"]),
        "height": int(box["height"]),
    }


def _split_two_chars(mask: np.ndarray) -> Optional[List[Tuple[np.ndarray, Dict[str, int]]]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    tight = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    projection = (tight > 0).sum(axis=0)
    empty = projection == 0
    runs = _empty_runs(empty)
    central_runs = [
        run for run in runs if 0.20 * tight.shape[1] <= (run[0] + run[1]) / 2.0 <= 0.80 * tight.shape[1]
    ]

    if central_runs:
        split_start, split_end, _ = max(central_runs, key=lambda run: run[2])
    else:
        mid = tight.shape[1] // 2
        split_start, split_end = mid, mid

    raw_parts = (
        (tight[:, :split_start], 0),
        (tight[:, split_end:], split_end),
    )
    parts: List[Tuple[np.ndarray, Dict[str, int]]] = []
    for part, x_offset in raw_parts:
        trimmed = _trim_mask(part, margin=2)
        if trimmed is None:
            return None
        trimmed_mask, local_box = trimmed
        box = {
            "x": int(xs.min() + x_offset + local_box["x"]),
            "y": int(ys.min() + local_box["y"]),
            "width": int(local_box["width"]),
            "height": int(local_box["height"]),
        }
        parts.append((trimmed_mask, box))

    return parts


def _empty_runs(empty: np.ndarray) -> List[Tuple[int, int, int]]:
    runs: List[Tuple[int, int, int]] = []
    start: Optional[int] = None
    for idx, value in enumerate(empty):
        if value and start is None:
            start = idx
        if (not value or idx == len(empty) - 1) and start is not None:
            end = idx if not value else idx + 1
            runs.append((start, end, end - start))
            start = None
    return runs


def _trim_mask(mask: np.ndarray, margin: int = 0) -> Optional[Tuple[np.ndarray, Dict[str, int]]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(mask.shape[1], int(xs.max()) + 1 + margin)
    y2 = min(mask.shape[0], int(ys.max()) + 1 + margin)
    return mask[y1:y2, x1:x2], {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def _recognize_char(mask: np.ndarray, allowed: Iterable[str], char_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    densities = _grid_densities(mask)
    scores = []
    for char in allowed:
        patterns = list(_PATTERNS[char])
        if char_index is not None:
            patterns.extend(_POSITIONAL_PATTERNS.get((char, char_index), []))
        char_score = max(_pattern_score(densities, pattern) for pattern in patterns)
        scores.append((char_score, char))

    if char_index == 0 and scores:
        best_score, best_char = max(scores)
        r_score = next((score for score, char in scores if char == "R"), None)
        if (
            best_char in {"F", "P"}
            and r_score is not None
            and best_score - r_score <= 0.04
            and _has_r_diagonal_leg(densities)
        ):
            scores = [
                (best_score + 0.001, char) if char == "R" else (score, char)
                for score, char in scores
            ]

    if char_index == 1 and scores:
        best_score, best_char = max(scores)
        five_score = next((score for score, char in scores if char == "5"), None)
        if (
            best_char == "J"
            and five_score is not None
            and best_score - five_score <= 0.015
            and _has_5_structure(densities)
        ):
            scores = [
                (best_score + 0.001, char) if char == "5" else (score, char)
                for score, char in scores
            ]

    scores.sort(reverse=True)
    if not scores:
        return None

    best_score, best_char = scores[0]
    return {
        "char": best_char,
        "score": round(float(best_score), 4),
        "alternatives": [
            {"char": char, "score": round(float(score), 4)}
            for score, char in scores[:5]
        ],
    }


def _grid_densities(mask: np.ndarray) -> np.ndarray:
    trimmed = _trim_mask(mask, margin=2)
    if trimmed is None:
        return np.zeros((_GRID_ROWS, _GRID_COLS), dtype=np.float32)

    mask = trimmed[0]
    densities = np.zeros((_GRID_ROWS, _GRID_COLS), dtype=np.float32)
    for row in range(_GRID_ROWS):
        y1 = int(row * mask.shape[0] / _GRID_ROWS)
        y2 = int((row + 1) * mask.shape[0] / _GRID_ROWS)
        for col in range(_GRID_COLS):
            x1 = int(col * mask.shape[1] / _GRID_COLS)
            x2 = int((col + 1) * mask.shape[1] / _GRID_COLS)
            cell = mask[y1:y2, x1:x2]
            densities[row, col] = float((cell > 0).mean()) if cell.size else 0.0
    return densities


def _row_projection_sharpness(mask: np.ndarray) -> float:
    projection = (mask > 0).sum(axis=1).astype(np.float32)
    return float(np.square(projection).sum() / max(float(projection.sum()), 1.0))


def _apply_dot_row_correction(mask: np.ndarray, correction: float) -> np.ndarray:
    if correction == 0.0:
        return mask
    height, width = mask.shape
    center_x = (width - 1) / 2.0
    padding = int(np.ceil(abs(correction) * width / 2.0)) + 3
    transform = np.array(
        [[1.0, 0.0, 0.0], [correction, 1.0, padding - correction * center_x]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        mask,
        transform,
        (width, height + 2 * padding),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )


def _estimate_dot_row_correction(masks: Iterable[np.ndarray]) -> float:
    masks = list(masks)
    baseline_sharpness = [_row_projection_sharpness(mask) for mask in masks]
    best_correction = 0.0
    best_ratio = 1.0

    for step in range(-6, 7):
        if step == 0:
            continue
        correction = step * 0.05
        sharpness_ratio = float(
            np.mean(
                [
                    _row_projection_sharpness(_apply_dot_row_correction(mask, correction))
                    / baseline
                    for mask, baseline in zip(masks, baseline_sharpness)
                ]
            )
        )
        if sharpness_ratio > best_ratio:
            best_correction = correction
            best_ratio = sharpness_ratio

    return best_correction if best_ratio >= 1.02 else 0.0


def _pattern_score(densities: np.ndarray, pattern: Tuple[str, ...]) -> float:
    template = np.array([[1 if value == "1" else 0 for value in row] for row in pattern], dtype=bool)
    positive = densities[template].mean() if template.any() else 0.0
    negative = densities[~template].mean() if (~template).any() else 0.0
    return float(positive - 0.65 * negative)


def _has_r_diagonal_leg(densities: np.ndarray) -> bool:
    leg_bands = (
        max(float(densities[4, 2]), float(densities[4, 3])),
        max(float(densities[5, 2]), float(densities[5, 3]), float(densities[5, 4])),
        max(float(densities[6, 3]), float(densities[6, 4])),
    )
    return min(leg_bands) >= 0.16 and float(np.mean(leg_bands)) >= 0.24


def _has_5_structure(densities: np.ndarray) -> bool:
    upper_left = float(densities[1:3, 0:2].mean())
    middle_bar = float(densities[3, 1:4].mean())
    lower_right = float(densities[4:6, 3:5].mean())
    bottom_bar = float(densities[6, 0:4].mean())
    return upper_left >= 0.18 and middle_bar >= 0.35 and lower_right >= 0.30 and bottom_bar >= 0.24


def _public_result(best: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    bbox = best["bbox"]
    char_boxes = []
    for box in best["char_boxes"]:
        char_boxes.append(
            {
                "x": bbox["x"] + box["x"],
                "y": bbox["y"] + box["y"],
                "width": box["width"],
                "height": box["height"],
            }
        )

    return {
        "found": True,
        "text": best["text"],
        "confidence": round(float(best["confidence"]), 3),
        "bbox": bbox,
        "char_boxes": char_boxes,
        "roi": best["roi"],
        "orientation": best.get("orientation", "normal"),
        "component_count": best["component_count"],
        "chars": best["chars"],
        "candidate_score": round(float(best["candidate_score"]), 4),
        "candidates": [
            {
                "text": item.get("text", ""),
                "roi": item.get("roi", ""),
                "orientation": item.get("orientation", "normal"),
                "bbox": item.get("bbox"),
                "confidence": round(float(item.get("confidence", 0.0)), 3),
                "candidate_score": round(float(item.get("candidate_score", 0.0)), 4),
            }
            for item in sorted(candidates, key=lambda candidate: candidate["candidate_score"], reverse=True)[:3]
        ],
    }


def _make_debug_images(gray: np.ndarray, best: Dict[str, Any]) -> Dict[str, np.ndarray]:
    bbox = best["bbox"]
    crop = _crop_with_margin(gray, bbox, margin=max(20, int(max(bbox["width"], bbox["height"]) * 0.20)))
    crop_vis = _stretch_to_bgr(crop)

    binary = best["mask"]
    binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    overview = _stretch_to_bgr(gray)
    max_dim = 2000
    scale = 1.0
    h, w = overview.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        overview = cv2.resize(overview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    x1 = int(bbox["x"] * scale)
    y1 = int(bbox["y"] * scale)
    x2 = int((bbox["x"] + bbox["width"]) * scale)
    y2 = int((bbox["y"] + bbox["height"]) * scale)
    cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        overview,
        f"Mark {best['text']}",
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for idx, char_box in enumerate(best.get("char_boxes", [])):
        cx1 = int((bbox["x"] + char_box["x"]) * scale)
        cy1 = int((bbox["y"] + char_box["y"]) * scale)
        cx2 = int((bbox["x"] + char_box["x"] + char_box["width"]) * scale)
        cy2 = int((bbox["y"] + char_box["y"] + char_box["height"]) * scale)
        cv2.rectangle(overview, (cx1, cy1), (cx2, cy2), (0, 180, 255), 1)
        if idx < len(best.get("text", "")):
            cv2.putText(
                overview,
                best["text"][idx],
                (cx1, max(18, cy1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 255),
                1,
                cv2.LINE_AA,
            )

    crop_vis = _resize_debug(crop_vis, max_side=720)
    binary_vis = _resize_debug(binary_vis, max_side=720, interpolation=cv2.INTER_NEAREST)
    return {"overview": overview, "crop": crop_vis, "binary": binary_vis}


def _crop_with_margin(gray: np.ndarray, bbox: Dict[str, int], margin: int) -> np.ndarray:
    x1 = max(0, bbox["x"] - margin)
    y1 = max(0, bbox["y"] - margin)
    x2 = min(gray.shape[1], bbox["x"] + bbox["width"] + margin)
    y2 = min(gray.shape[0], bbox["y"] + bbox["height"] + margin)
    return gray[y1:y2, x1:x2]


def _stretch_to_bgr(gray: np.ndarray) -> np.ndarray:
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    if max_val <= min_val:
        stretched = np.zeros_like(gray, dtype=np.uint8)
    else:
        stretched = ((gray.astype(np.float32) - min_val) * 255.0 / (max_val - min_val)).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def _resize_debug(image: np.ndarray, max_side: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) == 0:
        return image
    scale = max_side / max(h, w)
    if scale <= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=interpolation)
