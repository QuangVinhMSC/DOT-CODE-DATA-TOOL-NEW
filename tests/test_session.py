import os
import sys

import pytest

from dotgen.core import session
from dotgen.core.io_config import ConfigError
from dotgen.core.models import CharFormat, DotLink, Quad


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point every session path at a throwaway directory.

    Without this the suite would autosave into the developer's real
    %LOCALAPPDATA% and hand them a bogus recovery prompt on the next start.
    """
    root = tmp_path / "home"
    monkeypatch.setenv("DOTGEN_HOME", str(root))

    return root


def populate(state, dotted_image, backgrounds) -> None:
    """A small but multi-tab state, so the round trip is worth asserting on."""
    state.add_sample_image("orig.png", dotted_image)
    state.set_quad(0, Quad([(2, 2), (40, 3), (41, 30), (3, 29)]))
    state.set_param("dot.area", "max", 60.0)

    state.save_char_format(
        CharFormat("1", dots=[(0, 0), (0, 2)], links=[DotLink(0, 1, "v", 1.0)])
    )
    state.set_active_char("1")

    state.add_background("bg0.png", backgrounds[0])
    state.add_line()
    state.add_char(0, "1")
    state.set_export(out_dir="out", images_per_job=7, seed=99)


def raise_boom(*args, **kwargs):
    raise ConfigError("boom")


def temp_files(root) -> list[str]:
    return [n for n in os.listdir(root) if n.endswith(".tmp")]


# ----------------------------------------------------------------------
# locations
# ----------------------------------------------------------------------


def test_paths_honour_dotgen_home_and_are_created(home):
    assert not home.exists()

    assert os.path.abspath(session.app_dir()) == os.path.abspath(str(home))
    assert home.is_dir()

    assert session.session_path() == os.path.join(str(home), session.SESSION_NAME)
    assert session.log_dir() == os.path.join(str(home), session.LOG_DIR_NAME)
    assert os.path.isdir(session.log_dir())


def test_app_dir_falls_back_to_localappdata(tmp_path, monkeypatch):
    monkeypatch.delenv("DOTGEN_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert session.app_dir() == os.path.join(str(tmp_path), session.APP_DIR_NAME)


# ----------------------------------------------------------------------
# the session round trip
# ----------------------------------------------------------------------


def test_save_and_load_session_restores_state(home, state, app, dotted_image, backgrounds):
    from dotgen.core.state import AppState

    populate(state, dotted_image, backgrounds)

    path = session.save_session(state)

    assert path == session.session_path()
    assert os.path.isfile(path)

    dst = AppState()
    assert session.load_session(dst) is True

    assert len(dst.sample_images) == 1
    assert dst.quads[0].pts[1] == (40.0, 3.0)
    assert dst.params["dot.area"].max == 60.0
    assert dst.active_char == "1"
    assert dst.char_formats["1"].links[0].axis == "v"
    assert len(dst.backgrounds) == 1
    assert len(dst.lines) == 1
    assert dst.export.seed == 99


def test_load_session_overwrites_a_mutated_state(home, state, dotted_image, backgrounds):
    populate(state, dotted_image, backgrounds)
    session.save_session(state)

    state.set_param("dot.area", "max", 999.0)
    state.set_active_char("0")

    assert session.load_session(state) is True
    assert state.params["dot.area"].max == 60.0
    assert state.active_char == "1"


def test_load_session_without_a_file(home, state):
    assert session.load_session(state) is False


# ----------------------------------------------------------------------
# lifecycle
# ----------------------------------------------------------------------


def test_session_lifecycle(home, state, dotted_image, backgrounds):
    assert session.has_session() is False
    assert session.session_age() is None

    populate(state, dotted_image, backgrounds)
    session.save_session(state)

    assert session.has_session() is True

    age = session.session_age()
    assert age is not None and 0.0 <= age < 60.0

    session.clear_session()

    assert session.has_session() is False
    assert session.session_age() is None
    assert not os.path.exists(session.session_path())


def test_clear_session_is_quiet_when_there_is_nothing(home):
    session.clear_session()
    session.clear_session()


# ----------------------------------------------------------------------
# damaged files
# ----------------------------------------------------------------------


def test_corrupt_session_does_not_raise(home, state):
    with open(session.session_path(), "wb") as fh:
        fh.write(b"\x00 not a zip at all \xff" * 40)

    assert session.load_session(state) is False
    assert session.has_session() is False


def test_truncated_session_does_not_raise(home, state, dotted_image, backgrounds):
    populate(state, dotted_image, backgrounds)
    session.save_session(state)

    data = open(session.session_path(), "rb").read()

    with open(session.session_path(), "wb") as fh:
        fh.write(data[: len(data) // 2])

    assert session.load_session(state) is False


def test_empty_session_file_is_not_offered(home):
    open(session.session_path(), "wb").close()

    assert session.has_session() is False


# ----------------------------------------------------------------------
# the autosave must not raise, and must not corrupt what is already there
# ----------------------------------------------------------------------


def test_save_session_returns_none_instead_of_raising(home, state, monkeypatch):
    monkeypatch.setattr(session, "save_config", raise_boom)

    assert session.save_session(state) is None


def test_failed_save_leaves_the_previous_session_intact(
    home, state, monkeypatch, dotted_image, backgrounds
):
    populate(state, dotted_image, backgrounds)
    session.save_session(state)

    good = open(session.session_path(), "rb").read()

    monkeypatch.setattr(session, "save_config", raise_boom)

    assert session.save_session(state) is None
    assert open(session.session_path(), "rb").read() == good
    assert temp_files(home) == []


def test_successful_save_leaves_no_temp_file(home, state, dotted_image, backgrounds):
    populate(state, dotted_image, backgrounds)

    session.save_session(state)
    session.save_session(state)

    assert temp_files(home) == []
    assert os.listdir(home) == [session.SESSION_NAME]


def test_failed_save_writes_a_log(home, state, monkeypatch):
    monkeypatch.setattr(session, "save_config", raise_boom)
    session.save_session(state)

    logs = os.listdir(session.log_dir())

    assert len(logs) == 1
    assert "boom" in open(os.path.join(session.log_dir(), logs[0]), encoding="utf-8").read()


# ----------------------------------------------------------------------
# crash reporting
# ----------------------------------------------------------------------


def make_error() -> tuple:
    """A real exception with a real traceback, raised and caught on purpose."""

    def inner():
        raise ValueError("the printer is on fire")

    try:
        inner()
    except ValueError:
        return sys.exc_info()

    raise AssertionError("unreachable")


def test_format_crash_carries_context_and_traceback(home):
    text = session.format_crash(*make_error())

    assert "ValueError" in text
    assert "the printer is on fire" in text
    assert "Traceback (most recent call last)" in text
    assert "make_error" in text and "inner" in text

    assert "python" in text
    assert sys.version.splitlines()[0] in text
    assert "0.2.0" in text  # dotgen.__version__


def test_write_crash_log_writes_the_same_report(home):
    info = make_error()
    path = session.write_crash_log(*info)

    assert os.path.dirname(path) == session.log_dir()
    assert path.endswith(".log")

    text = open(path, encoding="utf-8").read()

    assert "ValueError" in text
    assert "the printer is on fire" in text
    assert "Traceback (most recent call last)" in text


def test_crash_logs_do_not_overwrite_each_other(home):
    info = make_error()
    paths = {session.write_crash_log(*info) for _ in range(5)}

    assert len(paths) == 5


def test_log_rotation_keeps_the_cap(home):
    info = make_error()

    for _ in range(session.MAX_LOGS + 8):
        session.write_crash_log(*info)

    assert len(os.listdir(session.log_dir())) == session.MAX_LOGS


def test_log_rotation_keeps_the_newest(home):
    info = make_error()
    kept = [session.write_crash_log(*info) for _ in range(session.MAX_LOGS + 3)]

    on_disk = {os.path.join(session.log_dir(), n) for n in os.listdir(session.log_dir())}

    assert on_disk == set(kept[-session.MAX_LOGS :])


# ----------------------------------------------------------------------
# the excepthook
# ----------------------------------------------------------------------


def test_install_excepthook_logs_calls_back_and_chains(home, monkeypatch):
    seen: list[tuple] = []
    chained: list[tuple] = []

    def previous(exc_type, exc, tb):
        chained.append((exc_type, exc, tb))

    monkeypatch.setattr(sys, "excepthook", previous)
    session.install_excepthook(lambda path, details: seen.append((path, details)))

    assert sys.excepthook is not previous

    info = make_error()
    sys.excepthook(*info)

    assert len(chained) == 1
    assert chained[0][1] is info[1]

    assert len(seen) == 1
    path, details = seen[0]

    assert path is not None and os.path.isfile(path)
    assert "the printer is on fire" in details
    assert open(path, encoding="utf-8").read() == details or "Traceback" in details


def test_install_excepthook_without_a_callback(home, monkeypatch):
    chained: list[tuple] = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: chained.append(a))

    session.install_excepthook()
    sys.excepthook(*make_error())

    assert len(chained) == 1
    assert len(os.listdir(session.log_dir())) == 1


def test_install_excepthook_twice_does_not_chain_to_itself(home, monkeypatch):
    chained: list[tuple] = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: chained.append(a))

    session.install_excepthook()
    session.install_excepthook()

    sys.excepthook(*make_error())

    # One trip to the original hook, not two and not a recursion.
    assert len(chained) == 1


def test_excepthook_survives_a_failing_callback(home, monkeypatch):
    chained: list[tuple] = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: chained.append(a))

    session.install_excepthook(raise_boom)
    sys.excepthook(*make_error())

    assert len(chained) == 1


def test_excepthook_survives_a_failing_log_write(home, monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(session, "write_crash_log", raise_boom)

    session.install_excepthook(lambda path, details: seen.append((path, details)))
    sys.excepthook(*make_error())

    assert seen[0][0] is None
    assert "the printer is on fire" in seen[0][1]


def test_excepthook_restores_cleanly(home, monkeypatch):
    original = sys.excepthook

    monkeypatch.setattr(sys, "excepthook", original)
    session.install_excepthook()
    monkeypatch.undo()

    assert sys.excepthook is original
