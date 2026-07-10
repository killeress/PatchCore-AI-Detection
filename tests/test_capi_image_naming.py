from capi_config import CAPIConfig
from capi_image_naming import (
    canonical_image_prefix,
    image_prefix_display_labels,
    panel_image_group_key,
    source_image_prefix,
)


def test_canonical_image_prefix_supports_legacy_and_hm_names():
    assert canonical_image_prefix("G0F00000_110454.tif") == "G0F00000"
    assert canonical_image_prefix("G0F00000083754.tif") == "G0F00000"
    assert canonical_image_prefix("R0F00000083753.tif") == "R0F00000"
    assert canonical_image_prefix("W0F00000083751.tif") == "W0F00000"
    assert canonical_image_prefix("WGF50500083752.tif") == "WGF50500"
    assert canonical_image_prefix("B0F00000083756.tif") == "B0F00000"
    assert canonical_image_prefix("U0F00000083755.tif") == "STANDARD"
    assert canonical_image_prefix("PINIGBI0083748.tif") == "PINIGBI0083748"


def test_source_image_prefix_preserves_hm_display_name():
    assert source_image_prefix("U0F00000083755.tif") == "U0F00000"
    assert source_image_prefix("STANDARD_110456.tif") == "STANDARD"


def test_image_prefix_display_labels_prefers_source_alias():
    assert image_prefix_display_labels([
        "STANDARD_110456.tif",
        "U0F00000083755.tif",
        "G0F00000083754.tif",
    ]) == {
        "STANDARD": "U0F00000",
        "G0F00000": "G0F00000",
    }


def test_hm_b0f_name_matches_existing_skip_file_config():
    cfg = CAPIConfig(skip_files=["B0F00000"])

    assert cfg.should_skip_file("B0F00000_110459.tif") is True
    assert cfg.should_skip_file("B0F00000083756.tif") is True


def test_panel_image_group_key_groups_retake_names():
    assert panel_image_group_key("G0F00000_110454.tif") == "G0F00000"
    assert panel_image_group_key("G0F00000083754.tif") == "G0F00000"
    assert panel_image_group_key("U0F00000083755.tif") == "STANDARD"
    assert panel_image_group_key("PINIGBI0083748.tif") == "PINIGBI"
