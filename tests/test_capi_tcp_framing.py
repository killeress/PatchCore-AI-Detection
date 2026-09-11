import pytest

from capi_server import _extract_request_frame, _legacy_request_ready, parse_request


BOMB_REQUEST = (
    b"AOI@T865QE48AJ55;GN140JCAL070S;CAPI39;1920,1200;NG;WGF50500;"
    b"(11/11;960/11);//192.168.1.66/images/T865QE48AJ55"
)
NEXT_REQUEST = b"AOI@NEXT;MODEL;CAPI39;1920,1200;OK;//images/NEXT"


def extract_chunks(chunks):
    pending = b""
    frames = []
    for chunk in chunks:
        pending += chunk
        while True:
            frame, pending, _ = _extract_request_frame(pending)
            if frame is None:
                break
            frames.append(frame)
    return frames, pending


@pytest.mark.parametrize("ending", [b"\r\n", b"\n", b"\r", b"\0"])
def test_every_split_position_preserves_frames_and_partial_tail(ending):
    stream = BOMB_REQUEST + NEXT_REQUEST + ending + BOMB_REQUEST[:60]
    for cut in range(len(stream) + 1):
        frames, pending = extract_chunks([stream[:cut], stream[cut:]])
        assert frames == [BOMB_REQUEST, NEXT_REQUEST], cut
        assert pending == BOMB_REQUEST[:60], cut


def test_one_byte_at_a_time_preserves_unicode_and_coordinate_semicolons():
    request = BOMB_REQUEST.replace(b"/images/", "/圖片/".encode())
    frames, pending = extract_chunks(bytes([b]) for b in request + b"\r\n")
    assert frames == [request]
    assert pending == b""
    parsed = parse_request(frames[0].decode())
    assert "/圖片/" in parsed["image_dir"]
    assert parsed["bomb_info"]["coordinates"] == [(11, 11), (960, 11)]


def test_terminated_frame_does_not_consume_next_partial_frame():
    frames, pending = extract_chunks([NEXT_REQUEST + b"\r\n" + BOMB_REQUEST[:59]])
    assert frames == [NEXT_REQUEST]
    assert pending == BOMB_REQUEST[:59]
    assert not _legacy_request_ready(pending)


def test_residue_does_not_hide_a_following_request_even_when_aoi_marker_is_split():
    frames, pending = extract_chunks([b"50500;(11/11);//old/pathAO", b"I@" + NEXT_REQUEST[4:] + b"\n"])
    assert frames == [b"50500;(11/11);//old/path", NEXT_REQUEST]
    assert pending == b""


@pytest.mark.parametrize("request_bytes", [
    BOMB_REQUEST,
    NEXT_REQUEST,
    b"AOI@G1;MODEL;CAPI39;1920,1200;OK;;;C:\\images\\different-id",
    b"AOI@G1;MODEL;CAPI39;1920,1200;HY;;();//images/G1",
    b"AOI@G1;MODEL;AAPI09;1920,1200;NG;//images/different-id;W0F00000,CDK2(123,456)",
])
def test_legacy_syntax_accepts_supported_request_shapes(request_bytes):
    assert _legacy_request_ready(request_bytes)


@pytest.mark.parametrize("tail", [
    b"WGF", b"WGF50500", b"WGF50500;", b"WGF50500;(11/11;960/",
    b"WGF50500;(11/11;960/11)", b"WGF50500;(11/11;960/11);",
    b";(11/11;960/", b"//images/", b"//images/\xe5\x9c",
])
def test_legacy_syntax_rejects_known_incomplete_bomb_path_and_utf8_fragments(tail):
    assert not _legacy_request_ready(b"AOI@G1;MODEL;CAPI39;1920,1200;NG;" + tail)


@pytest.mark.parametrize("tail", [b"", b";W0F", b";W0F00000,CDK2(12,", b";W0F00000,CDK2(12,34)W0F"])
def test_aapi_ng_waits_for_its_coordinate_payload(tail):
    assert not _legacy_request_ready(b"AOI@G1;MODEL;AAPI09;1920,1200;NG;//images/G1" + tail)
