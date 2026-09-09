import logging
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import capi_server


@pytest.fixture
def tcp_server(monkeypatch, caplog):
    monkeypatch.setattr(capi_server, "server_status", capi_server.ServerStatusTracker())
    monkeypatch.setattr(logging.getLogger("capi"), "propagate", True)
    caplog.set_level(logging.INFO, logger="capi.server")
    server = capi_server.CAPIServer.__new__(capi_server.CAPIServer)
    server.recv_timeout = 0
    server.recv_buffer_size = 4096
    server._ensure_auto_model_switch_for_request = MagicMock(return_value=SimpleNamespace())
    server._process_request = MagicMock(return_value=(
        "OK", "[]", [], False, None, None, False, None, None,
    ))
    server._queue_save_results_async = MagicMock()
    server._save_error_record = MagicMock()
    return server


def make_socket(chunks):
    client = MagicMock()
    client.getsockname.return_value = ("192.168.1.200", 7907)
    client.gettimeout.return_value = None
    client.recv.side_effect = chunks
    return client


def messages(caplog, tag):
    return [record.getMessage() for record in caplog.records if tag in record.getMessage()]


def test_send_logs_begin_before_write_and_completion_after_write(tcp_server, caplog, monkeypatch):
    client = make_socket([])
    response = "AOI@玻璃;OK\r\n@QJPG-玻璃;OK,"
    payload = (response + "\r\n").encode("utf-8")

    def send(data):
        assert data == payload
        assert messages(caplog, "[TCP_SEND_BEGIN]")
        assert not messages(caplog, "[TCP_SEND_OK]")

    client.sendall.side_effect = send
    monkeypatch.setattr(capi_server.time, "monotonic", MagicMock(side_effect=[10, 10.125]))
    capi_server._send_response(client, response, "conn=test req=2 Glass=玻璃")
    completed = messages(caplog, "[TCP_SEND_OK]")[0]
    assert f"bytes={len(payload)}" in completed
    assert "elapsed_ms=125.0" in completed
    assert "conn=test req=2 Glass=玻璃" in completed


@pytest.mark.parametrize("error", [ConnectionResetError(104, "reset"), socket.timeout("timed out")])
def test_send_failure_preserves_exception_and_records_duration(tcp_server, caplog, monkeypatch, error):
    client = make_socket([])
    client.sendall.side_effect = error
    monkeypatch.setattr(capi_server.time, "monotonic", MagicMock(side_effect=[20, 25]))
    with pytest.raises(type(error)) as caught:
        capi_server._send_response(client, "ERR", "conn=test req=1", kind="protocol_error")
    assert caught.value is error
    failed = messages(caplog, "[TCP_SEND_FAILED]")[0]
    assert "elapsed_ms=5000.0" in failed
    assert f"errno={error.errno}" in failed
    assert "kind=protocol_error" in failed
    assert any(record.exc_info for record in caplog.records)
    assert not messages(caplog, "[TCP_SEND_OK]")


@pytest.mark.parametrize("judgment,kind", [("OK", "result"), ("HY", "hy")])
def test_handler_correlates_repeated_glass_requests_and_eof(tcp_server, caplog, judgment, kind):
    request = f"AOI@G1;MODEL;CAPI33;1920,1200;{judgment};/images\r\n".encode()
    client = make_socket([request, request, b""])
    tcp_server._handle_client(client, ("192.168.1.3", 20245))
    assert client.sendall.call_count == 2
    assert tcp_server._queue_save_results_async.call_count == 2
    completed = messages(caplog, "[TCP_SEND_OK]")
    assert len(completed) == 2
    for number, message in enumerate(completed, 1):
        assert f"req={number} Glass=G1 Machine=CAPI33 kind={kind}" in message
        assert "peer=('192.168.1.3', 20245)" in message
    conn_id = messages(caplog, "[TCP_OPEN]")[0].split("conn=")[1].split()[0]
    assert all(f"conn={conn_id}" in message for message in completed)
    assert "reason=peer_eof requests_received=2 handled=2" in messages(caplog, "[TCP_CLOSE]")[0]
    assert messages(caplog, "[TCP_RECV_EOF]")
    client.close.assert_called_once()
    client.settimeout.assert_called_once_with(None)


@pytest.mark.parametrize("request_bytes,error,kind", [
    (b"bad request\n", None, "protocol_error"),
    (b"AOI@G1;MODEL;CAPI33;1920,1200;OK;/images\n", RuntimeError("inference failed"), "internal_error"),
])
def test_error_responses_also_have_send_logs(tcp_server, caplog, request_bytes, error, kind):
    tcp_server._process_request.side_effect = error
    client = make_socket([request_bytes, b""])
    tcp_server._handle_client(client, ("192.168.1.3", 20245))
    assert f"kind={kind}" in messages(caplog, "[TCP_SEND_OK]")[0]
    tcp_server._save_error_record.assert_called_once()


def test_failed_result_and_error_reply_are_both_logged(tcp_server, caplog):
    client = make_socket([b"AOI@G1;MODEL;CAPI33;1920,1200;OK;/images\n"])
    client.sendall.side_effect = ConnectionResetError(104, "reset")
    tcp_server._handle_client(client, ("192.168.1.3", 20245))
    failed = messages(caplog, "[TCP_SEND_FAILED]")
    assert len(failed) == 2
    assert "kind=result" in failed[0]
    assert "kind=internal_error" in failed[1]
    assert all("req=1 Glass=G1 Machine=CAPI33" in message for message in failed)
    assert not messages(caplog, "[TCP_SEND_OK]")
    assert "reason=error_response_send_failed" in messages(caplog, "[TCP_CLOSE]")[0]
    tcp_server._queue_save_results_async.assert_not_called()


@pytest.mark.parametrize("end,tag,reason", [
    (b"", "TCP_RECV_EOF", "peer_eof"),
    (socket.timeout("timed out"), "TCP_RECV_TIMEOUT", "recv_timeout"),
    (ConnectionResetError(104, "reset"), "TCP_SOCKET_ERROR", "socket_error"),
])
def test_partial_receive_then_disconnect_or_timeout(tcp_server, caplog, end, tag, reason):
    client = make_socket([b"AOI@", end])
    tcp_server._handle_client(client, ("192.168.1.3", 20245))
    assert "bytes=4 buffered_bytes=4" in messages(caplog, "[TCP_RECV]")[0]
    assert messages(caplog, f"[{tag}]")
    assert f"reason={reason}" in messages(caplog, "[TCP_CLOSE]")[0]
    assert not messages(caplog, "[TCP_REQUEST]")
    client.sendall.assert_not_called()
