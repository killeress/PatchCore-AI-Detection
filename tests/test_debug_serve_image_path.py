import os

from capi_web import CAPIWebHandler


def test_debug_image_path_resolves_newest_w0f00000_image_from_folder(tmp_path):
    folder = tmp_path / "panel"
    folder.mkdir()
    older = folder / "W0F00000_older.tif"
    newer = folder / "prefix_W0F00000_newer.PNG"
    nested = folder / "nested"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    nested.mkdir()
    (nested / "W0F00000_nested.tif").write_bytes(b"nested")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    assert CAPIWebHandler._resolve_debug_image_path(folder) == newer


def test_debug_image_path_ignores_side_images_starting_with_s(tmp_path):
    folder = tmp_path / "panel"
    folder.mkdir()
    front = folder / "W0F00000_front.tif"
    upper_side = folder / "SW0F00000_side.tif"
    lower_side = folder / "side_W0F00000_latest.tif"
    front.write_bytes(b"front")
    upper_side.write_bytes(b"upper side")
    lower_side.write_bytes(b"lower side")
    os.utime(front, ns=(1_000_000_000, 1_000_000_000))
    os.utime(upper_side, ns=(2_000_000_000, 2_000_000_000))
    os.utime(lower_side, ns=(3_000_000_000, 3_000_000_000))

    assert CAPIWebHandler._resolve_debug_image_path(folder) == front


def test_debug_image_path_keeps_file_and_rejects_folder_without_w0f00000(tmp_path):
    image = tmp_path / "other.jpg"
    image.write_bytes(b"image")
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()

    assert CAPIWebHandler._resolve_debug_image_path(image) == image
    assert CAPIWebHandler._resolve_debug_image_path(empty_folder) is None


def test_debug_serve_image_uses_resolved_folder_image(tmp_path):
    folder = tmp_path / "panel"
    folder.mkdir()
    selected = folder / "capture_W0F00000_latest.png"
    selected.write_bytes(b"image")

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    served = []
    errors = []
    handler._send_binary = lambda path: served.append(path)
    handler._send_error = lambda code, message, path="": errors.append((code, message, path))
    handler._inference_rotate_180_enabled = lambda: False

    handler._handle_debug_serve_image({"path": [str(folder)]})

    assert served == [str(selected)]
    assert errors == []


def test_debug_serve_image_can_return_resolved_file_path(tmp_path):
    folder = tmp_path / "panel"
    folder.mkdir()
    selected = folder / "W0F00000_latest.png"
    selected.write_bytes(b"image")

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    payloads = []
    served = []
    handler._send_json = lambda payload, **kwargs: payloads.append(payload)
    handler._send_binary = lambda path: served.append(path)

    handler._handle_debug_serve_image({"path": [str(folder)], "resolve": ["1"]})

    assert payloads == [{"image_path": str(selected)}]
    assert served == []
