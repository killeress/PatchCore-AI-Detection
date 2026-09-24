"""Coexisting CAPI screens must not share images, coordinates, or models."""
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from capi_config import CAPIConfig
from capi_inference import AOIReportDefect, CAPIInferencer
from capi_preprocess import filter_panel_lighting_files
from capi_station_adapter import create_station_adapter
from capi_web import CAPIWebHandler, _within_spec_screen_code


@pytest.mark.parametrize("newer", ["U0F00000", "STANDARD"])
@pytest.mark.parametrize("retake", [False, True])
def test_panel_selection_preserves_both_screens_and_deduplicates_each(tmp_path, newer, retake):
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("capi")
    worker.config = SimpleNamespace(max_images_per_panel=20)
    expected = {}
    for screen in ("U0F00000", "STANDARD"):
        path = tmp_path / f"{screen}085943.tif"
        path.write_bytes(b"image")
        os.utime(path, (20 if screen == newer else 10,) * 2)
        expected[screen] = path
        if retake:
            path = tmp_path / f"{screen}_085944.tif"
            path.write_bytes(b"retake")
            os.utime(path, (30, 30))
            expected[screen] = path
    selected, duplicate = worker._prepare_panel_image_files(tmp_path)
    assert set(selected) == set(expected.values())
    assert duplicate is retake
    assert filter_panel_lighting_files(tmp_path) == expected
    for screen, path in expected.items():
        assert worker.station_adapter.find_lighting_image(tmp_path, screen) == path


def test_coexisting_screens_use_independent_models_and_thresholds():
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("capi")
    worker.config = CAPIConfig(
        is_new_architecture=True, machine_id="TEST",
        threshold_mapping={"U0F00000": {"inner": 0.2, "edge": 0.3},
                           "STANDARD": {"inner": 0.8, "edge": 0.9}},
    )
    worker.threshold = 0.75
    worker._get_model_for = Mock(side_effect=lambda _machine, screen, zone: (screen, zone))
    for screen, thresholds in worker.config.threshold_mapping.items():
        prefix = worker._get_image_prefix(f"{screen}085943.tif")
        for zone, threshold in thresholds.items():
            assert worker._get_inferencer_for_zone(prefix, zone) == (screen, zone)
            assert worker._get_threshold_for_zone(prefix, zone) == threshold


@pytest.mark.parametrize("screen,mapping", [
    ("U0F00000", {"STANDARD": "standard.pt"}),
    ("STANDARD", {"U0F00000": "u0f.pt"}),
])
def test_legacy_inference_does_not_fall_back_between_independent_screens(screen, mapping):
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("capi")
    worker._model_mapping = mapping
    worker.inferencer = object()
    with pytest.raises(RuntimeError, match=screen + ".*model_mapping"):
        worker._get_inferencer_for_prefix(screen)


def test_legacy_u0f_model_load_failure_does_not_fall_back_to_default():
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("capi")
    worker._model_mapping = {"U0F00000": "u0f.pt"}
    worker._inferencers = {}
    worker._load_model_from_path = Mock(return_value=None)
    worker.inferencer = object()
    with pytest.raises(RuntimeError, match="U0F00000.*model_mapping"):
        worker._get_inferencer_for_prefix("U0F00000")


@pytest.mark.parametrize("bomb_screen,report_screen", [
    ("U0F00000", "STANDARD"), ("STANDARD", "U0F00000"),
])
def test_other_screen_aoi_does_not_cover_client_bomb(bomb_screen, report_screen):
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("capi")
    worker.config = CAPIConfig(bomb_area_force_detection_enabled=True)
    report = {report_screen: [AOIReportDefect("PCDK2", 350, 230, report_screen)]}
    updated, added = worker._aoi_report_with_forced_client_bomb_coords(
        report, dict(image_prefix=bomb_screen, defect_type="point", coordinates=[(350, 230)]),
    )
    assert added == 1
    assert set(updated) == {bomb_screen, report_screen}
    assert len(updated[bomb_screen]) == len(updated[report_screen]) == 1


def test_new_training_detects_both_screens_with_independent_labels(tmp_path):
    for screen in ("U0F00000", "STANDARD"):
        (tmp_path / f"{screen}_085943.tif").write_bytes(b"image")
    adapter = create_station_adapter("capi")
    detected = CAPIWebHandler._detect_train_new_lightings([tmp_path], adapter)
    assert detected == ["U0F00000", "STANDARD"]
    labels = CAPIWebHandler._train_new_lighting_labels(detected, [tmp_path], adapter)
    assert labels == {"U0F00000": "U0F00000", "STANDARD": "STANDARD"}
    units = CAPIWebHandler._all_train_unit_labels(SimpleNamespace(station_adapter=adapter))
    assert all(f"{screen}-{zone}" in units for screen in detected for zone in ("inner", "edge"))


def test_within_spec_and_aoi_numbering_keep_both_screens_independent():
    adapter = create_station_adapter("capi")
    screens = CAPIConfig().within_spec_judgment_rules["default"]["screens"]
    for screen in ("U0F00000", "STANDARD"):
        assert _within_spec_screen_code(f"{screen}085943.tif", screens, adapter) == screen
    # The same product coordinate is row 1 on U0F and row 2 on STANDARD.
    detail = {"aoi_machine_coords": {
        "U0F00000": [{"product_x": 10, "product_y": 20}],
        "STANDARD": [{"product_x": 1, "product_y": 2}, {"product_x": 10, "product_y": 20}],
    }, "images": [
        {"image_name": f"{screen}_085943.tif", "tiles": [
            dict(tile_id=0, is_aoi_coord=True, aoi_product_x=10, aoi_product_y=20),
        ]} for screen in ("U0F00000", "STANDARD")
    ]}
    CAPIWebHandler._decorate_record_aoi_point_numbers(detail, adapter)
    assert [img["tiles"][0]["aoi_point_number"] for img in detail["images"]] == [1, 2]


def test_legacy_standard_ng_cache_with_u0f_source_is_not_reused(tmp_path):
    from capi_train_new import sample_ng_tiles

    db = Mock()
    db.list_training_bomb_validation_samples.return_value = [dict(
        lighting="STANDARD", zone="inner", image_name="U0F00000085943.tif",
        crop_path=str(tmp_path / "old.png"),
    )]
    db.list_training_bomb_candidates.return_value = []
    stats = sample_ng_tiles(
        job_id="j1", machine_id="M", over_review_root=tmp_path,
        db=db, thumb_dir=tmp_path / "thumbs", lightings=("STANDARD",),
        ng_validation_base_dir=tmp_path, log=lambda _msg: None,
    )
    assert stats["invalid_skipped"] == 1
    assert stats["sampled"] == 0
    db.insert_tile_pool.assert_not_called()
