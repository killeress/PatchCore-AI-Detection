from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_debug_exclusion_transfer_requires_and_sends_product_code():
    template = (ROOT / "templates" / "debug_inference.html").read_text(
        encoding="utf-8"
    )

    assert 'id="cv-product-code"' in template
    assert "if (!productCode)" in template
    assert "product=${encodeURIComponent(productCode)}" in template


def test_debug_coordinate_overlay_is_sized_after_viewer_becomes_visible():
    template = (ROOT / "templates" / "debug_inference.html").read_text(
        encoding="utf-8"
    )
    loader = template.split("function cvLoadImage", 1)[1].split(
        "function cvSyncOverlay", 1
    )[0]

    show_viewer = "classList.remove('hidden')"
    assert loader.index(show_viewer) < loader.index("cvSyncOverlay()")


def test_settings_exclusion_transfer_consumes_product_code():
    template = (ROOT / "templates" / "settings.html").read_text(encoding="utf-8")

    assert "params.get('product')" in template
    assert "modelResolutionMap[exInitialProductCode]" in template


def test_settings_exclusion_transfer_restores_selection_after_image_load():
    template = (ROOT / "templates" / "settings.html").read_text(encoding="utf-8")
    loader = template.split("function exLoadImage", 1)[1].split(
        "function exResetToSample", 1
    )[0]

    assert "preserveSelection" in loader
    assert "selectionToRestore" in loader
    assert "exCurRect = selectionToRestore" in loader
