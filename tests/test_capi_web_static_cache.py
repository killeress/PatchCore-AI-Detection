import io

from capi_web import CAPIWebHandler


def _capture_binary_headers(path):
    handler = object.__new__(CAPIWebHandler)
    headers = {}
    handler.send_response = lambda _code: None
    handler.send_header = lambda name, value: headers.__setitem__(name, value)
    handler.end_headers = lambda: None
    handler.wfile = io.BytesIO()
    handler._send_404 = lambda: (_ for _ in ()).throw(AssertionError("404"))
    handler._send_binary(str(path))
    return headers


def test_send_binary_disables_cache_for_code_files(tmp_path):
    for name in ("a.html", "b.js", "c.mjs", "d.css"):
        asset = tmp_path / name
        asset.write_text("x", encoding="utf-8")
        assert _capture_binary_headers(asset)["Cache-Control"] == "no-cache", name


def test_send_binary_keeps_day_cache_for_assets(tmp_path):
    png = tmp_path / "banner.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert _capture_binary_headers(png)["Cache-Control"] == "max-age=86400"
