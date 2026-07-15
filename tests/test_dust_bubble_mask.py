from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


def _make_inferencer(*, detect_bubbles: bool) -> CAPIInferencer:
    config = CAPIConfig()
    config.dust_area_min = 5
    config.dust_area_max = 100000
    config.dust_extension = 0
    config.dust_detect_dark_particles = False
    config.dust_detect_bubbles_enabled = detect_bubbles

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    return inferencer


def _low_contrast_bubble_image() -> np.ndarray:
    image = np.full((128, 128), 80, dtype=np.uint8)
    cv2.ellipse(image, (64, 64), (18, 24), 0, 0, 360, 72, -1)
    return cv2.GaussianBlur(image, (9, 9), 0)


def _low_contrast_bright_bubble_image() -> np.ndarray:
    image = np.full((128, 128), 60, dtype=np.uint8)
    cv2.ellipse(image, (64, 64), (18, 24), 0, 0, 360, 68, -1)
    return cv2.GaussianBlur(image, (9, 9), 0)


def _low_contrast_ring_bubble_image() -> np.ndarray:
    image = np.full((128, 128), 80, dtype=np.uint8)
    cv2.ellipse(image, (64, 64), (22, 28), 0, 0, 360, 90, 5)
    return cv2.GaussianBlur(image, (9, 9), 0)


def _striped_low_contrast_bubble_image() -> np.ndarray:
    image = np.full((512, 512), 80, dtype=np.int16)
    stripes = ((np.arange(512) % 2) * 2 - 1) * 4
    image += stripes[None, :]

    bubble = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(bubble, (270, 335), (45, 40), 0, 0, 360, 255, -1)
    image[bubble > 0] -= 6

    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


def _striped_broken_ring_bubble_image() -> np.ndarray:
    image = np.full((512, 512), 80, dtype=np.int16)
    stripes = ((np.arange(512) % 2) * 2 - 1) * 4
    image += stripes[None, :]

    bubble = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(bubble, (270, 335), (45, 40), 0, 0, 360, 255, 8)
    cv2.rectangle(bubble, (250, 305), (310, 335), 0, -1)
    image[bubble > 0] -= 6

    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


def _striped_surface_edge_bubble_image() -> np.ndarray:
    image = np.full((512, 512), 70, dtype=np.int16)
    stripes = ((np.arange(512) % 2) * 2 - 1) * 3
    image += stripes[None, :]
    image[376:, :] += 35

    bubble = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(bubble, (270, 335), (45, 25), 0, 0, 360, 255, -1)
    image[bubble > 0] += 6

    lower_blob = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(lower_blob, (330, 445), (35, 25), 0, 0, 360, 255, -1)
    image[lower_blob > 0] += 4

    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


def _surface_edge_partial_bubble_image() -> np.ndarray:
    image = np.full((512, 512), 34, dtype=np.int16)
    stripes = ((np.arange(512) % 2) * 2 - 1) * 2
    image += stripes[None, :]
    image[376:, :] += 70

    bubble = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(bubble, (430, 347), (31, 29), 0, 0, 360, 255, -1)
    rows = np.arange(512)[:, None]
    image[(bubble > 0) & (rows < 350)] += 6
    image[(bubble > 0) & (rows >= 350)] += 1

    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


def _large_rotated_surface_bubble_image() -> np.ndarray:
    yy, xx = np.mgrid[:512, :512]
    image = (
        np.full((512, 512), 31, dtype=np.float32)
        + xx * 0.02
        + yy * 0.005
        + ((xx % 2) * 2 - 1) * 2
    )

    bubble = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(bubble, (300, 337), (190, 75), -32, 0, 360, 255, -1)
    image[bubble > 0] += 18
    image[440:, :] += 100

    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


def _large_rotated_bubble_panel_image() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:700, :900]
    image = (
        np.full((700, 900), 31, dtype=np.float32)
        + xx * 0.02
        + yy * 0.005
        + ((xx % 2) * 2 - 1) * 2
    )

    bubble = np.zeros((700, 900), dtype=np.uint8)
    cv2.ellipse(bubble, (500, 417), (190, 75), -32, 0, 360, 255, -1)
    image[bubble > 0] += 18
    image[520:, :] += 100

    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0), bubble


def _regular_boundary_bubble_panel_image() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((700, 900), 80, dtype=np.uint8)

    interior_bubble = np.zeros_like(image)
    cv2.ellipse(interior_bubble, (300, 250), (25, 30), 0, 0, 360, 255, -1)
    image[interior_bubble > 0] = 72

    boundary_bubble = np.zeros_like(image)
    cv2.ellipse(boundary_bubble, (600, 330), (30, 35), 0, 0, 360, 255, -1)
    image[boundary_bubble > 0] = 72

    return cv2.GaussianBlur(image, (9, 9), 0), boundary_bubble


def test_low_contrast_bubble_mask_is_opt_in():
    image = _low_contrast_bubble_image()

    inferencer = _make_inferencer(detect_bubbles=False)
    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    assert is_dust is False
    assert np.count_nonzero(mask) == 0
    assert ratio == 0.0
    assert "Bub:" not in detail


def test_low_contrast_bubble_is_added_to_dust_mask_when_enabled():
    image = _low_contrast_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    yy, xx = np.ogrid[:128, :128]
    expected_bubble = ((xx - 64) ** 2 / (18 ** 2) + (yy - 64) ** 2 / (24 ** 2)) <= 1
    covered = np.count_nonzero(mask[expected_bubble] > 0) / np.count_nonzero(expected_bubble)

    assert is_dust is True
    assert covered >= 0.60
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_low_contrast_bright_bubble_is_added_to_dust_mask_when_enabled():
    image = _low_contrast_bright_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    yy, xx = np.ogrid[:128, :128]
    expected_bubble = ((xx - 64) ** 2 / (18 ** 2) + (yy - 64) ** 2 / (24 ** 2)) <= 1
    covered = np.count_nonzero(mask[expected_bubble] > 0) / np.count_nonzero(expected_bubble)

    assert is_dust is True
    assert covered >= 0.60
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_low_contrast_ring_bubble_fills_interior_when_enabled():
    image = _low_contrast_ring_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    yy, xx = np.ogrid[:128, :128]
    expected_bubble = ((xx - 64) ** 2 / (22 ** 2) + (yy - 64) ** 2 / (28 ** 2)) <= 1
    covered = np.count_nonzero(mask[expected_bubble] > 0) / np.count_nonzero(expected_bubble)

    assert is_dust is True
    assert covered >= 0.75
    assert mask[64, 64] == 255
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_striped_low_contrast_bubble_is_detected_when_enabled():
    image = _striped_low_contrast_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    yy, xx = np.ogrid[:512, :512]
    expected_bubble = ((xx - 270) ** 2 / (45 ** 2) + (yy - 335) ** 2 / (40 ** 2)) <= 1
    covered = np.count_nonzero(mask[expected_bubble] > 0) / np.count_nonzero(expected_bubble)

    assert is_dust is True
    assert covered >= 0.70
    assert mask[335, 270] == 255
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_striped_broken_ring_bubble_fills_interior_when_enabled():
    image = _striped_broken_ring_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    yy, xx = np.ogrid[:512, :512]
    expected_bubble = ((xx - 270) ** 2 / (45 ** 2) + (yy - 335) ** 2 / (40 ** 2)) <= 1
    covered = np.count_nonzero(mask[expected_bubble] > 0) / np.count_nonzero(expected_bubble)

    assert is_dust is True
    assert covered >= 0.70
    assert mask[335, 270] == 255
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_surface_edge_blob_below_boundary_is_not_bubble():
    image = _striped_surface_edge_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    assert is_dust is True
    assert mask[335, 270] == 255
    assert mask[445, 330] == 0
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_surface_edge_partial_bubble_fills_to_boundary():
    image = _surface_edge_partial_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    yy, xx = np.ogrid[:512, :512]
    lower_bubble = (
        ((xx - 430) ** 2 / (31 ** 2) + (yy - 347) ** 2 / (29 ** 2)) <= 1
    ) & (yy >= 352) & (yy < 368)
    covered = np.count_nonzero(mask[lower_bubble] > 0) / np.count_nonzero(lower_bubble)

    assert is_dust is True
    assert covered >= 0.75
    assert mask[365, 430] == 255
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_large_rotated_surface_bubble_is_filled():
    image = _large_rotated_surface_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    expected_bubble = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(expected_bubble, (300, 337), (190, 75), -32, 0, 360, 255, -1)
    covered = (
        np.count_nonzero((mask > 0) & (expected_bubble > 0))
        / np.count_nonzero(expected_bubble)
    )

    assert is_dust is True
    assert covered >= 0.85
    assert mask[337, 300] == 255
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_large_rectangular_surface_glare_is_not_bubble():
    yy, xx = np.mgrid[:512, :512]
    image = (
        np.full((512, 512), 31, dtype=np.float32)
        + xx * 0.02
        + yy * 0.005
        + ((xx % 2) * 2 - 1) * 2
    )
    image[220:440, 135:465] += 18
    image[440:, :] += 100
    image = cv2.GaussianBlur(np.clip(image, 0, 255).astype(np.uint8), (5, 5), 0)

    inferencer = _make_inferencer(detect_bubbles=True)
    _, mask, _, detail = inferencer.check_dust_or_scratch_feature(image)

    assert mask[330, 300] == 0
    assert "Bub:0" in detail


def test_padded_context_recovers_boundary_clipped_large_bubble():
    panel_image, bubble = _large_rotated_bubble_panel_image()
    inferencer = _make_inferencer(detect_bubbles=True)
    tile_x, tile_y, tile_size = 100, 80, 512
    omit_crop = panel_image[
        tile_y:tile_y + tile_size,
        tile_x:tile_x + tile_size,
    ]
    expected = bubble[
        tile_y:tile_y + tile_size,
        tile_x:tile_x + tile_size,
    ]

    _, baseline_mask, _, baseline_detail = \
        inferencer.check_dust_or_scratch_feature(omit_crop)
    is_dust, mask, ratio, detail = \
        inferencer._check_dust_or_scratch_feature_with_context(
            panel_image,
            tile_x,
            tile_y,
            tile_size,
            tile_size,
            omit_crop,
            focus_x=500,
        )

    baseline_covered = (
        np.count_nonzero((baseline_mask > 0) & (expected > 0))
        / np.count_nonzero(expected)
    )
    covered = (
        np.count_nonzero((mask > 0) & (expected > 0))
        / np.count_nonzero(expected)
    )

    assert "Bub:0" in baseline_detail
    assert baseline_covered < 0.20
    assert is_dust is True
    assert covered >= 0.90
    assert mask[337, 400] == 255
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_context_recovers_regular_boundary_bubble_when_another_bubble_exists():
    panel_image, boundary_bubble = _regular_boundary_bubble_panel_image()
    inferencer = _make_inferencer(detect_bubbles=True)
    tile_x, tile_y, tile_size = 100, 80, 512
    omit_crop = panel_image[
        tile_y:tile_y + tile_size,
        tile_x:tile_x + tile_size,
    ]
    expected = boundary_bubble[
        tile_y:tile_y + tile_size,
        tile_x:tile_x + tile_size,
    ]

    _, baseline_mask, _, baseline_detail = \
        inferencer.check_dust_or_scratch_feature(omit_crop)
    is_dust, mask, ratio, detail = \
        inferencer._check_dust_or_scratch_feature_with_context(
            panel_image,
            tile_x,
            tile_y,
            tile_size,
            tile_size,
            omit_crop,
            focus_x=500,
        )

    baseline_covered = (
        np.count_nonzero((baseline_mask > 0) & (expected > 0))
        / np.count_nonzero(expected)
    )
    covered = (
        np.count_nonzero((mask > 0) & (expected > 0))
        / np.count_nonzero(expected)
    )

    assert "Bub:1" in baseline_detail
    assert baseline_covered < 0.20
    assert is_dust is True
    assert covered >= 0.85
    assert mask[250, 500] == 255
    assert ratio > 0.0
    assert "Bub:2" in detail
    assert "CtxShift:96" in detail


def test_bubble_detector_ignores_small_dark_specks():
    image = np.full((128, 128), 80, dtype=np.uint8)
    image[60:64, 60:64] = 72
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    assert is_dust is False
    assert np.count_nonzero(mask) == 0
    assert ratio == 0.0
    assert "Bub:0" in detail


def test_bubble_detector_setting_can_hot_reload_from_db():
    config = CAPIConfig()

    config.apply_db_overrides([
        {"param_name": "dust_detect_bubbles_enabled", "decoded_value": "true"},
    ])

    assert config.dust_detect_bubbles_enabled is True
    assert config.to_dict()["dust_detect_bubbles_enabled"] is True


def test_settings_api_marks_dataclass_bool_params_as_bool():
    from capi_web import CAPIWebHandler

    captured = {}
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = SimpleNamespace(get_all_config_params=lambda: [])
    handler.inferencer = SimpleNamespace(config=CAPIConfig())
    handler._current_settings_user = lambda: None
    handler._send_json = lambda payload: captured.update(payload)

    handler._handle_api_settings()

    params = {p["param_name"]: p for p in captured["params"]}
    assert params["dust_detect_bubbles_enabled"]["param_type"] == "bool"


def test_settings_api_normalizes_existing_db_bool_param_type():
    from capi_web import CAPIWebHandler

    captured = {}
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = SimpleNamespace(get_all_config_params=lambda: [{
        "param_name": "dust_detect_bubbles_enabled",
        "param_value": '"True"',
        "param_type": "str",
        "decoded_value": "True",
    }])
    handler.inferencer = SimpleNamespace(config=CAPIConfig())
    handler._current_settings_user = lambda: None
    handler._send_json = lambda payload: captured.update(payload)

    handler._handle_api_settings()

    params = {p["param_name"]: p for p in captured["params"]}
    assert params["dust_detect_bubbles_enabled"]["param_type"] == "bool"


def test_settings_api_hides_removed_pixel_grid_mask_ratio():
    from capi_web import CAPIWebHandler

    captured = {}
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = SimpleNamespace(get_all_config_params=lambda: [{
        "param_name": "dust_pixel_grid_max_mask_ratio",
        "param_value": "0.15",
        "param_type": "float",
        "decoded_value": 0.15,
    }])
    handler.inferencer = SimpleNamespace(config=CAPIConfig())
    handler._current_settings_user = lambda: None
    handler._send_json = lambda payload: captured.update(payload)

    handler._handle_api_settings()

    names = {p["param_name"] for p in captured["params"]}
    assert "dust_pixel_grid_max_mask_ratio" not in names


def test_bool_update_repairs_legacy_string_param_type():
    from capi_database import CAPIDatabase

    db_path = Path(f"_test_bool_param_repair_{uuid.uuid4().hex}.db")
    db = CAPIDatabase(str(db_path))
    try:
        conn = db._get_conn()
        try:
            conn.execute(
                """INSERT INTO config_params
                   (param_name, param_value, param_type, description)
                   VALUES (?, ?, ?, ?)""",
                ("dust_detect_bubbles_enabled", '"True"', "str", ""),
            )
            conn.commit()
        finally:
            conn.close()

        db.update_config_param(
            "dust_detect_bubbles_enabled",
            False,
            reason="unit test",
        )
        updated = db.get_config_param("dust_detect_bubbles_enabled")

        assert updated["param_type"] == "bool"
        assert updated["decoded_value"] is False
    finally:
        db_path.unlink(missing_ok=True)


def test_1366_pixel_grid_filter_uses_configured_gaussian_only_for_target_resolution(monkeypatch):
    inferencer = _make_inferencer(detect_bubbles=False)
    inferencer.config.dust_pixel_grid_filter_enabled = True
    inferencer.config.dust_pixel_grid_blur_kernel = 7
    image = np.full((64, 64), 80, dtype=np.uint8)

    original_blur = cv2.GaussianBlur
    blur_kernels = []

    def record_blur(src, ksize, sigma_x, *args, **kwargs):
        blur_kernels.append(ksize)
        return original_blur(src, ksize, sigma_x, *args, **kwargs)

    monkeypatch.setattr(cv2, "GaussianBlur", record_blur)

    inferencer.check_dust_or_scratch_feature(
        image,
        product_resolution=(1366, 768),
    )
    assert blur_kernels == [(7, 7), (7, 7)]

    blur_kernels.clear()
    inferencer.check_dust_or_scratch_feature(
        image,
        product_resolution=(1920, 1080),
    )
    assert blur_kernels == []


def test_pixel_grid_filter_settings_can_hot_reload_from_db():
    config = CAPIConfig()

    config.apply_db_overrides([
        {"param_name": "dust_pixel_grid_filter_enabled", "decoded_value": "true"},
        {"param_name": "dust_pixel_grid_blur_kernel", "decoded_value": 9},
    ])

    assert config.dust_pixel_grid_filter_enabled is True
    assert config.dust_pixel_grid_blur_kernel == 9
    serialized = config.to_dict()
    assert serialized["dust_pixel_grid_filter_enabled"] is True
    assert serialized["dust_pixel_grid_blur_kernel"] == 9


def test_pixel_grid_toggle_rerenders_dependent_setting_locks():
    settings_html = (Path(__file__).resolve().parent.parent / "templates" / "settings.html").read_text(
        encoding="utf-8"
    )

    lock_set_start = settings_html.index("const LOCK_AFFECTING_BOOL_PARAMS")
    lock_set_end = settings_html.index("]);", lock_set_start)
    lock_set = settings_html[lock_set_start:lock_set_end]
    assert "'dust_pixel_grid_filter_enabled'" in lock_set
