import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from capi_web import CAPIWebHandler


@pytest.fixture
def sop_server(tmp_path):
    class Handler(CAPIWebHandler):
        jinja_env = None
        _capi_server_instance = SimpleNamespace(
            server_config={"sop": {"published_dir": "published"}},
            server_config_path=str(tmp_path / "server_config.yaml"),
        )

    Handler.init_jinja()
    root = tmp_path / "published"
    root.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def get(path):
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    yield root, Handler._capi_server_instance.server_config, get
    server.shutdown()
    server.server_close()
    thread.join()


def test_entry_defaults_and_explicit_center(sop_server):
    _, config, get = sop_server
    status, headers, _ = get("/help")
    assert status == 302
    assert headers["Location"] == "http://10.174.37.81/help/sop"
    config["mes_report"] = {"facility": "MOD1"}
    assert get("/help")[1]["Location"] == "http://10.172.25.105/help/sop"
    config["sop"]["center_url"] = "https://sop.internal:8443/"
    assert get("/help")[1]["Location"] == "https://sop.internal:8443/help/sop"
    # Serving the center itself must not redirect back through /help.
    assert get("/help/sop")[0] == 200


@pytest.mark.parametrize("url", ["javascript:alert(1)", "//host", "http://host/\r\nX:1", "http://host/help", "http://user:password@host"])
def test_invalid_center_url(sop_server, url):
    _, config, get = sop_server
    config["sop"]["center_url"] = url
    assert get("/help")[0] == 503


def test_publish_read_download_replace_and_remove(sop_server):
    root, _, get = sop_server
    folder = root / "日常操作"
    folder.mkdir()
    pdf = folder / "開機 SOP & V1.PDF"
    pdf.write_bytes(b"%PDF-1.4\nfirst")
    (folder / "draft.pptx").write_bytes(b"private")
    status, headers, body = get("/help/sop")
    assert status == 200
    page = body.decode()
    assert "開機 SOP &amp; V1" in page
    assert "日常操作" in page
    assert "draft.pptx" not in page
    assert "no-store" in headers["Cache-Control"]
    url = "/help/file?" + urlencode({"name": "日常操作/開機 SOP & V1.PDF"})
    status, headers, body = get(url)
    assert status == 200 and body == pdf.read_bytes()
    assert headers["Content-Type"] == "application/pdf"
    assert headers["Content-Disposition"].startswith("inline;")
    assert get(url + "&download=1")[1]["Content-Disposition"].startswith("attachment;")
    pdf.write_bytes(b"%PDF-1.4\nupdated")
    assert get(url)[2] == pdf.read_bytes()
    pdf.unlink()
    assert get(url)[0] == 404
    assert "目前尚無" in get("/help/sop")[2].decode()


@pytest.mark.parametrize("name", ["../secret.pdf", "..\\secret.pdf", "draft.pptx", "missing.pdf", "\x00.pdf"])
def test_reject_unpublished_paths(sop_server, name):
    root, _, get = sop_server
    (root.parent / "secret.pdf").write_bytes(b"secret")
    (root / "draft.pptx").write_bytes(b"private")
    assert get("/help/file?" + urlencode({"name": name}))[0] == 404


def test_outside_absolute_path_and_symlink(sop_server):
    root, _, get = sop_server
    secret = root.parent / "secret.pdf"
    secret.write_bytes(b"secret")
    assert get("/help/file?" + urlencode({"name": str(secret)}))[0] == 404
    try:
        (root / "link.pdf").symlink_to(secret)
    except OSError:
        pytest.skip("Symlink creation not permitted on this host")
    assert get("/help/file?name=link.pdf")[0] == 404
    assert "link.pdf" not in get("/help/sop")[2].decode()


def test_missing_directory_and_absolute_directory(sop_server):
    root, config, get = sop_server
    config["sop"]["published_dir"] = str(root)
    assert get("/help/sop")[0] == 200
    root.rmdir()
    status, _, body = get("/help/sop")
    assert status == 503
    assert "文件資料夾尚未就緒" in body.decode()


def test_help_assets_in_release():
    from scripts.build_deploy_zip import CODE_FILES
    for name in ("templates/help.html", "templates/dashboard_v3.html", "docs/sop_center.zh-TW.md"):
        assert name in CODE_FILES
        assert Path(name).is_file()
