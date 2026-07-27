import threading
from types import SimpleNamespace

from capi_server import CAPIServer


def test_start_marks_server_running_before_cleanup_scheduler_starts():
    observed_running_states = []
    server = SimpleNamespace(
        training_only=True,
        server_config={"web": {"enabled": False}},
        _running=False,
        _stop_requested=False,
        _web_server=None,
        _web_thread=None,
    )

    def start_cleanup_scheduler():
        observed_running_states.append(server._running)
        server._running = False

    server._start_cleanup_scheduler = start_cleanup_scheduler

    CAPIServer.start(server)

    assert observed_running_states == [True]


def test_start_aborts_when_stop_arrives_during_startup():
    server = SimpleNamespace(
        training_only=True,
        server_config={"web": {"enabled": False}},
        _running=False,
        _stop_requested=True,
    )
    server._start_cleanup_scheduler = lambda: (_ for _ in ()).throw(
        AssertionError("cleanup scheduler must not start after stop request")
    )

    CAPIServer.start(server)

    assert server._running is False


def test_cleanup_scheduler_thread_stops_with_server():
    server = SimpleNamespace(
        server_config={
            "cleanup": {
                "enabled": True,
                "schedule_time": "02:00",
                "vacuum_after_cleanup": False,
            }
        },
        _running=True,
        _cleanup_stop_event=threading.Event(),
        _cleanup_thread=None,
        _async_executor_lock=threading.Lock(),
        _async_executor_shutdown=True,
        _server_socket=None,
    )

    cleanup_thread = CAPIServer._start_cleanup_scheduler(server)

    assert cleanup_thread is server._cleanup_thread
    assert cleanup_thread.is_alive()

    CAPIServer.stop(server)

    assert not cleanup_thread.is_alive()
    assert server._cleanup_thread is None
