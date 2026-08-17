"""Crop 512x512 NG tiles from an Excel coordinate list.

The coordinate mapping intentionally reuses the current CAPI implementation:
raw object bounds provide the linear mapping, and a detected panel polygon is
used to correct points that fall outside the polygon or when the raw bounds
are contaminated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import openpyxl


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capi_inference import CAPIInferencer  # noqa: E402
from capi_preprocess import resolve_inward_polygon_tile  # noqa: E402


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_PRODUCT_RESOLUTION = (1920, 1200)
DEFAULT_TILE_SIZE = 512
DEFAULT_OTSU_OFFSET = 5


@dataclass(frozen=True)
class ExcelRecord:
    excel_row: int
    panel_id: str
    product_x: int
    product_y: int


@dataclass
class BoundaryInfo:
    raw_bounds: Tuple[int, int, int, int]
    otsu_bounds: Tuple[int, int, int, int]
    polygon: Optional[np.ndarray]
    polygon_area_ratio: float


def _normalise_header(value: Any) -> str:
    return re.sub(r"[\s_]+", "", str(value or "")).lower()


def _find_column(headers: Sequence[Any], kind: str) -> Optional[int]:
    normalised = [_normalise_header(value) for value in headers]
    if kind == "panel":
        for index, value in enumerate(normalised):
            if value in {"panelid", "panel"} or "panelid" in value:
                return index
    elif kind == "x":
        for index, value in enumerate(normalised):
            if value == "x" or value.startswith("x座標") or value.startswith("xcoordinate") or value.startswith("x"):
                return index
    elif kind == "y":
        for index, value in enumerate(normalised):
            if value == "y" or value.startswith("y座標") or value.startswith("ycoordinate") or value.startswith("y"):
                return index
    return None


def _as_int(value: Any, field_name: str, excel_row: int) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Excel row {excel_row}: {field_name} is empty")
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Excel row {excel_row}: invalid {field_name}={value!r}") from exc


def read_excel_records(excel_path: Path) -> List[ExcelRecord]:
    workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        headers = [cell.value for cell in worksheet[1]]
        panel_col = _find_column(headers, "panel")
        x_col = _find_column(headers, "x")
        y_col = _find_column(headers, "y")
        if panel_col is None or x_col is None or y_col is None:
            raise ValueError(
                "Excel must contain Panel ID, X coordinate, and Y coordinate columns; "
                f"headers={headers!r}"
            )

        records: List[ExcelRecord] = []
        for row_number, row in enumerate(worksheet.iter_rows(min_row=2, values_only=True), start=2):
            if not row or all(value is None or str(value).strip() == "" for value in row):
                continue
            panel_value = row[panel_col] if panel_col < len(row) else None
            if panel_value is None or str(panel_value).strip() == "":
                continue
            panel_id = str(panel_value).strip()
            records.append(
                ExcelRecord(
                    excel_row=row_number,
                    panel_id=panel_id,
                    product_x=_as_int(row[x_col], "X coordinate", row_number),
                    product_y=_as_int(row[y_col], "Y coordinate", row_number),
                )
            )
        return records
    finally:
        workbook.close()


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _shrink_polygon(polygon: np.ndarray, offset: int) -> np.ndarray:
    """Match calculate_otsu_bounds()'s legacy polygon inward offset."""
    result = np.asarray(polygon, dtype=np.float32).copy()
    if offset == 0:
        return result
    center_x = float(result[:, 0].mean())
    center_y = float(result[:, 1].mean())
    for index in range(len(result)):
        dx = float(result[index, 0]) - center_x
        dy = float(result[index, 1]) - center_y
        length = float(np.hypot(dx, dy))
        if length > 1e-6:
            result[index, 0] -= dx * offset / length
            result[index, 1] -= dy * offset / length
    return result


def detect_boundary(image: np.ndarray, tile_size: int, otsu_offset: int) -> BoundaryInfo:
    """Reuse the current raw-bound and polygon fitting path from capi_inference."""
    gray = _to_gray(image)
    helper = SimpleNamespace(config=SimpleNamespace(tile_size=tile_size))
    raw_bounds, binary_mask = CAPIInferencer._find_raw_object_bounds(helper, gray)
    if raw_bounds is None:
        raw_bounds = (0, 0, image.shape[1], image.shape[0])
    raw_bounds = tuple(int(value) for value in raw_bounds)

    polygon = CAPIInferencer._find_panel_polygon(helper, binary_mask, raw_bounds)
    if polygon is not None:
        polygon = _shrink_polygon(polygon, otsu_offset)

    height, width = gray.shape[:2]
    x1, y1, x2, y2 = raw_bounds
    otsu_bounds = (
        max(0, x1 + otsu_offset),
        max(0, y1 + otsu_offset),
        min(width, x2 - otsu_offset),
        min(height, y2 - otsu_offset),
    )
    if otsu_bounds[0] >= otsu_bounds[2] or otsu_bounds[1] >= otsu_bounds[3]:
        otsu_bounds = (0, 0, width, height)

    raw_area = max(1, (x2 - x1) * (y2 - y1))
    polygon_area_ratio = (
        abs(float(cv2.contourArea(polygon))) / raw_area if polygon is not None else 0.0
    )
    return BoundaryInfo(
        raw_bounds=raw_bounds,
        otsu_bounds=tuple(int(value) for value in otsu_bounds),
        polygon=polygon,
        polygon_area_ratio=polygon_area_ratio,
    )


def _linear_map(
    product_x: int,
    product_y: int,
    raw_bounds: Tuple[int, int, int, int],
    product_resolution: Tuple[int, int],
) -> Tuple[int, int]:
    x1, y1, x2, y2 = raw_bounds
    product_width, product_height = product_resolution
    return (
        int(product_x * (x2 - x1) / product_width + x1),
        int(product_y * (y2 - y1) / product_height + y1),
    )


def map_coordinate(
    record: ExcelRecord,
    boundary: BoundaryInfo,
    product_resolution: Tuple[int, int],
) -> Tuple[int, int, int, int, str, float]:
    linear_x, linear_y = _linear_map(
        record.product_x,
        record.product_y,
        boundary.raw_bounds,
        product_resolution,
    )
    mapped_x, mapped_y = CAPIInferencer._map_aoi_coords(
        None,
        record.product_x,
        record.product_y,
        boundary.raw_bounds,
        product_resolution,
        panel_polygon=boundary.polygon,
    )
    if boundary.polygon is None:
        mode = "linear_no_polygon"
        polygon_distance = 0.0
    else:
        mode = "polygon_corrected" if (mapped_x, mapped_y) != (linear_x, linear_y) else "linear_inside_polygon"
        polygon_distance = float(
            cv2.pointPolygonTest(
                np.asarray(boundary.polygon, dtype=np.float32),
                (float(mapped_x), float(mapped_y)),
                True,
            )
        )
    return linear_x, linear_y, int(mapped_x), int(mapped_y), mode, polygon_distance


def crop_bounds(
    center_x: int,
    center_y: int,
    image_width: int,
    image_height: int,
    tile_size: int,
    raw_bounds: Optional[Tuple[int, int, int, int]] = None,
    polygon: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, int, int, int]:
    if image_width < tile_size or image_height < tile_size:
        raise ValueError(
            f"image {image_width}x{image_height} is smaller than {tile_size}x{tile_size}"
        )
    half = tile_size // 2
    centered_x1 = center_x - half
    centered_y1 = center_y - half
    x1 = min(max(0, centered_x1), image_width - tile_size)
    y1 = min(max(0, centered_y1), image_height - tile_size)
    if polygon is not None and raw_bounds is not None:
        shift_axes = CAPIInferencer._resolve_aoi_inward_shift_axes(
            center_x,
            center_y,
            raw_bounds,
            tile_size,
        )
        x1, y1, _coverage, _shifted = resolve_inward_polygon_tile(
            anchor_xy=(center_x, center_y),
            polygon=polygon,
            image_shape=(image_height, image_width),
            tile_size=tile_size,
            initial_origin=(x1, y1),
            keep_anchor_inside=True,
            shift_axes=shift_axes,
        )
    return x1, y1, x1 + tile_size, y1 + tile_size, x1 - centered_x1, y1 - centered_y1


def _display_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image.copy()


def _write_image(path: Path, image: np.ndarray, extension: str = ".png") -> None:
    """Write through Python's Unicode-aware file API on Windows."""
    encoded_ok, encoded = cv2.imencode(extension, image)
    if not encoded_ok:
        raise IOError(f"failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def _put_text_with_background(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 1.2,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 3,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(8, min(x, max(8, image.shape[1] - text_width - 8)))
    y = max(text_height + baseline + 8, min(y, image.shape[0] - 8))
    cv2.rectangle(
        image,
        (x - 6, y - text_height - baseline - 6),
        (x + text_width + 6, y + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_overview(
    image: np.ndarray,
    boundary: BoundaryInfo,
    mapped_records: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    overview = _display_bgr(image)
    x1, y1, x2, y2 = boundary.raw_bounds
    cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 255, 255), 6)
    if boundary.polygon is not None:
        polygon_int = np.round(boundary.polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overview, [polygon_int], True, (255, 180, 0), 8, cv2.LINE_AA)

    for index, item in enumerate(mapped_records, start=1):
        crop_x1, crop_y1, crop_x2, crop_y2 = item["crop_bounds"]
        image_x, image_y = item["image_x"], item["image_y"]
        overlay = overview.copy()
        cv2.rectangle(overlay, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.14, overview, 0.86, 0, overview)
        cv2.rectangle(overview, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), 10)
        cv2.drawMarker(
            overview,
            (image_x, image_y),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            100,
            8,
            cv2.LINE_AA,
        )
        cv2.circle(overview, (image_x, image_y), 18, (0, 255, 0), 5, cv2.LINE_AA)
        label = (
            f"NG#{index} ExcelRow={item['excel_row']} "
            f"P=({item['product_x']},{item['product_y']}) "
            f"IMG=({image_x},{image_y})"
        )
        _put_text_with_background(overview, label, (crop_x1 + 10, max(55, crop_y1 - 15)))

    legend = (
        f"raw_bounds={boundary.raw_bounds}  "
        f"polygon={'yes' if boundary.polygon is not None else 'no'}  "
        f"area_ratio={boundary.polygon_area_ratio:.3f}"
    )
    _put_text_with_background(overview, legend, (20, 55), scale=1.1, color=(255, 255, 255))
    _write_image(output_path, overview)


def _image_index(input_dir: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            result.setdefault(path.stem.lower(), path)
    return result


def process(
    input_dir: Path,
    excel_path: Path,
    output_dir: Path,
    product_resolution: Tuple[int, int],
    tile_size: int,
    otsu_offset: int,
) -> Dict[str, Any]:
    records = read_excel_records(excel_path)
    images = _image_index(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "crops"
    overview_dir = output_dir / "overviews"
    crop_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[ExcelRecord]] = {}
    for record in records:
        grouped.setdefault(record.panel_id, []).append(record)

    for panel_id, panel_records in grouped.items():
        image_path = images.get(panel_id.lower())
        if image_path is None:
            for record in panel_records:
                rows.append(
                    {
                        "excel_row": record.excel_row,
                        "panel_id": record.panel_id,
                        "product_x": record.product_x,
                        "product_y": record.product_y,
                        "status": "missing_image",
                        "error": "no image with matching Panel ID",
                    }
                )
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            for record in panel_records:
                rows.append(
                    {
                        "excel_row": record.excel_row,
                        "panel_id": record.panel_id,
                        "product_x": record.product_x,
                        "product_y": record.product_y,
                        "image_path": str(image_path),
                        "status": "read_error",
                        "error": "cv2.imread returned None",
                    }
                )
            continue

        try:
            boundary = detect_boundary(image, tile_size, otsu_offset)
            mapped_items: List[Dict[str, Any]] = []
            for item_index, record in enumerate(panel_records, start=1):
                linear_x, linear_y, image_x, image_y, mapping_mode, polygon_distance = map_coordinate(
                    record, boundary, product_resolution
                )
                x1, y1, x2, y2, shift_x, shift_y = crop_bounds(
                    image_x,
                    image_y,
                    image.shape[1],
                    image.shape[0],
                    tile_size,
                    raw_bounds=boundary.raw_bounds,
                    polygon=boundary.polygon,
                )
                crop_path = crop_dir / (
                    f"{record.panel_id}_row{record.excel_row:02d}_"
                    f"P{record.product_x}_{record.product_y}_"
                    f"IMG{image_x}_{image_y}.png"
                )
                crop = image[y1:y2, x1:x2].copy()
                if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                    raise ValueError(f"unexpected crop shape {crop.shape}")
                _write_image(crop_path, crop)

                mapped_items.append(
                    {
                        "excel_row": record.excel_row,
                        "panel_id": record.panel_id,
                        "product_x": record.product_x,
                        "product_y": record.product_y,
                        "linear_image_x": linear_x,
                        "linear_image_y": linear_y,
                        "image_x": image_x,
                        "image_y": image_y,
                        "mapping_mode": mapping_mode,
                        "polygon_distance_px": polygon_distance,
                        "crop_bounds": (x1, y1, x2, y2),
                        "crop_shift_x": shift_x,
                        "crop_shift_y": shift_y,
                        "crop_path": str(crop_path),
                    }
                )

            overview_path = overview_dir / f"{panel_id}_overview.png"
            draw_overview(image, boundary, mapped_items, overview_path)
            for item in mapped_items:
                rows.append(
                    {
                        **item,
                        "image_path": str(image_path),
                        "image_width": image.shape[1],
                        "image_height": image.shape[0],
                        "raw_bounds": boundary.raw_bounds,
                        "otsu_bounds": boundary.otsu_bounds,
                        "polygon_detected": boundary.polygon is not None,
                        "polygon_area_ratio": boundary.polygon_area_ratio,
                        "overview_path": str(overview_path),
                        "status": "processed",
                        "error": "",
                    }
                )
        except Exception as exc:
            for record in panel_records:
                rows.append(
                    {
                        "excel_row": record.excel_row,
                        "panel_id": record.panel_id,
                        "product_x": record.product_x,
                        "product_y": record.product_y,
                        "image_path": str(image_path),
                        "status": "processing_error",
                        "error": str(exc),
                    }
                )

    csv_fields = [
        "excel_row", "panel_id", "product_x", "product_y", "status", "error",
        "image_path", "image_width", "image_height", "raw_bounds", "otsu_bounds",
        "polygon_detected", "polygon_area_ratio", "linear_image_x", "linear_image_y",
        "image_x", "image_y", "mapping_mode", "polygon_distance_px", "crop_bounds",
        "crop_shift_x", "crop_shift_y", "crop_path", "overview_path",
    ]
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_excel": str(excel_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "product_resolution": list(product_resolution),
        "tile_size": tile_size,
        "otsu_offset": otsu_offset,
        "excel_record_count": len(records),
        "processed_count": sum(row.get("status") == "processed" for row in rows),
        "missing_image_count": sum(row.get("status") == "missing_image" for row in rows),
        "error_count": sum(row.get("status") not in {"processed", "missing_image"} for row in rows),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "AAPI")
    parser.add_argument("--excel", type=Path, default=REPO_ROOT / "AAPI" / "NG資料統計.xlsx")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "AAPI" / "NG資料統計_512x512_結果",
    )
    parser.add_argument("--product-width", type=int, default=DEFAULT_PRODUCT_RESOLUTION[0])
    parser.add_argument("--product-height", type=int, default=DEFAULT_PRODUCT_RESOLUTION[1])
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--otsu-offset", type=int, default=DEFAULT_OTSU_OFFSET)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = process(
        input_dir=args.input_dir,
        excel_path=args.excel,
        output_dir=args.output_dir,
        product_resolution=(args.product_width, args.product_height),
        tile_size=args.tile_size,
        otsu_offset=args.otsu_offset,
    )
    print(json.dumps({key: summary[key] for key in (
        "source_excel", "output_dir", "product_resolution", "tile_size",
        "excel_record_count", "processed_count", "missing_image_count", "error_count",
    )}, ensure_ascii=False, indent=2))
    for row in summary["rows"]:
        if row.get("status") != "processed":
            print(
                f"{row.get('status')}: Excel row {row.get('excel_row')} "
                f"{row.get('panel_id')}: {row.get('error')}"
            )
    return 0 if summary["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
