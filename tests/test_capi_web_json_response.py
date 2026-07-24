import gzip
import io
import json

from capi_web import CAPIWebHandler


def _response_handler(accept_encoding: str = ""):
    handler = object.__new__(CAPIWebHandler)
    handler.headers = {"Accept-Encoding": accept_encoding}
    handler.wfile = io.BytesIO()
    handler.response_status = None
    handler.response_headers = {}
    handler.send_response = lambda status: setattr(handler, "response_status", status)
    handler.send_header = lambda name, value: handler.response_headers.__setitem__(name, value)
    handler.end_headers = lambda: None
    return handler


def test_send_json_can_use_compact_gzip_with_server_timing():
    handler = _response_handler("br, gzip")
    records = [{"glass_id": f"PANEL-{index:05d}"} for index in range(100)]

    metrics = handler._send_json(
        {"success": True, "records": records},
        compact=True,
        compress=True,
        server_timing={"sqlite": 0.012, "oracle": 1.234},
    )

    raw = gzip.decompress(handler.wfile.getvalue())
    assert json.loads(raw) == {
        "success": True,
        "records": records,
    }
    assert b"\n" not in raw
    assert handler.response_headers["Content-Encoding"] == "gzip"
    assert handler.response_headers["Vary"] == "Accept-Encoding"
    assert "sqlite;dur=12.0" in handler.response_headers["Server-Timing"]
    assert "oracle;dur=1234.0" in handler.response_headers["Server-Timing"]
    assert "json;dur=" in handler.response_headers["Server-Timing"]
    assert "gzip;dur=" in handler.response_headers["Server-Timing"]
    assert metrics["compressed"] is True
    assert metrics["response_bytes"] < metrics["uncompressed_bytes"]


def test_send_json_keeps_default_pretty_uncompressed_response():
    handler = _response_handler()

    metrics = handler._send_json({"success": True})

    assert handler.wfile.getvalue().startswith(b"{\n")
    assert "Content-Encoding" not in handler.response_headers
    assert metrics["compressed"] is False
