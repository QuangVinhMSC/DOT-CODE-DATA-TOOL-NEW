"""Crash survival: the autosaved session and the crash log.

A run of this tool is expensive to reproduce.  Sampling dots, drawing the six
formats of a character set and framing every background is an hour of careful
mouse work that lives only in :class:`~dotgen.core.state.AppState` until the
user remembers to save.  So the application writes that state out for itself,
periodically and on a clean exit, and offers it back on the next start.

The session file is an ordinary ``.dotcfg`` written by
:func:`~dotgen.core.io_config.save_config`.  Reusing the real format rather
than inventing a lighter one is deliberate: a recovered session is exactly as
complete as a file the user saved by hand, and every future field added to the
config travels into the autosave for free.

Two properties this module exists to guarantee:

*An autosave never takes down the program it is protecting.*  :func:`save_session`
catches everything -- a locked file, a full disk, an image that will not encode
-- reports the failure by returning ``None`` and writes the reason to the log
directory.  A timer that fires every minute is the last place a raised
exception should be allowed to escape.

*A crash mid-write cannot poison the next start.*  The archive is built under a
temporary name in the same directory and then moved onto the session path with
:func:`os.replace`, which is atomic on Windows as well as POSIX.  A process
killed halfway through leaves the previous good session where it was; it can
never leave a truncated zip that makes the next start offer a recovery it
cannot honour.  :func:`load_session` is defensive for the same reason: a
:class:`~dotgen.core.io_config.ConfigError` from a damaged file is an expected
outcome, not a bug, and turns into ``False``.

Every path goes through :func:`app_dir`, which honours the ``DOTGEN_HOME``
environment variable before it looks at ``%LOCALAPPDATA%``.  That override is
the seam the tests hang on -- without it a test run would autosave into the
developer's real profile directory and hand them somebody else's recovery
prompt the next morning.

Nothing here imports Qt.  The 60-second timer and the crash dialog belong to
the window; everything underneath them is plain standard library, so it runs
headless and under pytest.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime
from typing import Callable, TYPE_CHECKING

from .io_config import load_config, save_config

if TYPE_CHECKING:
    from .state import AppState

# The directory name under %LOCALAPPDATA%, and the variable that replaces the
# whole path.  DOTGEN_HOME *is* the application directory, not its parent.
APP_DIR_NAME = "dotgen"
HOME_ENV = "DOTGEN_HOME"

SESSION_NAME = "session.dotcfg"
LOG_DIR_NAME = "logs"

# How often the window should call save_session.  It lives here rather than in
# the timer so the policy and the thing it protects stay in one file.
AUTOSAVE_SECONDS = 60

# Crash logs are small, but a tool driven for hours can produce a stream of
# them; keep a working history and drop the rest.
MAX_LOGS = 20

_TEMP_PREFIX = ".session-"
_TEMP_SUFFIX = ".tmp"

__all__ = [
    "AUTOSAVE_SECONDS",
    "app_dir",
    "clear_session",
    "format_crash",
    "has_session",
    "install_excepthook",
    "load_session",
    "log_dir",
    "save_session",
    "session_age",
    "session_path",
    "write_crash_log",
]


# ----------------------------------------------------------------------
# locations
# ----------------------------------------------------------------------


def _ensure(path: str) -> str:
    """Create *path* if it is missing and return it either way.

    A path helper that raises is a path helper every caller has to wrap, so a
    failure to create the directory is left for the write that follows to
    report with a message about what it was actually trying to do.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass

    return path


def app_dir() -> str:
    """The per-user directory holding the session file and the logs.

    ``DOTGEN_HOME`` wins if it is set.  Otherwise this is
    ``%LOCALAPPDATA%/dotgen``; the ``APPDATA`` and home-directory fallbacks
    only matter off Windows, where the tests and the headless export path run.
    """
    override = os.environ.get(HOME_ENV)

    if override:
        return _ensure(os.path.abspath(override))

    base = (
        os.environ.get("LOCALAPPDATA")
        or os.environ.get("APPDATA")
        or os.path.expanduser("~")
    )

    return _ensure(os.path.join(os.path.abspath(base), APP_DIR_NAME))


def session_path() -> str:
    """Where the autosave lives."""
    return os.path.join(app_dir(), SESSION_NAME)


def log_dir() -> str:
    """Where crash logs are written."""
    return _ensure(os.path.join(app_dir(), LOG_DIR_NAME))


# ----------------------------------------------------------------------
# the session
# ----------------------------------------------------------------------


def save_session(state: "AppState") -> str | None:
    """Autosave *state*, atomically; return the path, or ``None`` on failure.

    The archive is written beside its destination and then moved onto it, so
    an interrupted save leaves the previous session untouched rather than a
    half-written zip.  Nothing propagates out of here: the caller is a timer
    or a close handler, and neither has anything useful to do with an
    exception.
    """
    target = session_path()
    tmp = None

    try:
        fd, tmp = tempfile.mkstemp(
            prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX, dir=os.path.dirname(target)
        )
        os.close(fd)

        save_config(state, tmp)
        os.replace(tmp, target)

        return target

    except Exception:
        _log_current_exception()

        # The temp file is ours alone; a leftover would accumulate once a
        # minute for as long as the failure lasts.
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass

        return None


def load_session(state: "AppState") -> bool:
    """Restore the autosaved session into *state*; ``False`` if it cannot be.

    :func:`~dotgen.core.io_config.load_config` overwrites *state* field by
    field, so a file that turns out to be damaged partway through leaves it
    partly replaced.  Recovery is therefore a startup operation on a fresh
    :class:`~dotgen.core.state.AppState`; do not offer it over work the user
    has already done in this run.
    """
    path = session_path()

    if not os.path.isfile(path):
        return False

    try:
        load_config(state, path)
    except Exception:
        _log_current_exception()
        return False

    return True


def has_session() -> bool:
    """Whether a session file exists that is worth offering to recover.

    Obvious rubbish -- an empty file, or one that is not a zip at all -- is
    rejected here so the user is never asked about a recovery that
    :func:`load_session` would then have to refuse.
    """
    path = session_path()

    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False

        return zipfile.is_zipfile(path)
    except OSError:
        return False


def session_age() -> float | None:
    """Seconds since the session file was written, or ``None`` if there is none.

    The recovery prompt uses this to say how old the offered work is; a
    session from four seconds ago reads very differently from one from
    yesterday.
    """
    try:
        return max(0.0, time.time() - os.path.getmtime(session_path()))
    except OSError:
        return None


def clear_session() -> None:
    """Delete the session file, if there is one.

    Called after a recovery has succeeded and after the user declines one:
    in both cases the file has served its purpose, and leaving it would offer
    the same stale work again on the next start.
    """
    try:
        os.remove(session_path())
    except OSError:
        pass


# ----------------------------------------------------------------------
# crash reporting
# ----------------------------------------------------------------------


def format_crash(exc_type, exc, tb) -> str:
    """The full crash report as text -- header, then the traceback.

    This is what the dialog's ``Copy details`` button puts on the clipboard
    and what :func:`write_crash_log` writes to disk, deliberately the same
    string: a report pasted into an issue should carry the same context as
    the file on the user's machine, which is why the interpreter and platform
    details are in here rather than only in the log header.
    """
    lines = [
        "DotGen crash report",
        f"time      : {datetime.now().isoformat(timespec='seconds')}",
        f"version   : {_dotgen_version()}",
        f"python    : {sys.version.splitlines()[0]}",
        f"platform  : {platform.platform()}",
        f"executable: {sys.executable}",
        "",
    ]

    try:
        body = "".join(traceback.format_exception(exc_type, exc, tb))
    except Exception:
        # A broken __repr__ on the exception must not cost us the report.
        body = f"{getattr(exc_type, '__name__', exc_type)}: <unformattable traceback>\n"

    return "\n".join(lines) + body


def write_crash_log(exc_type, exc, tb) -> str:
    """Write a timestamped crash log into :func:`log_dir` and return its path."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    base = os.path.join(log_dir(), f"crash-{stamp}")
    path = f"{base}.log"

    # The clock is coarse enough on Windows that two crashes in a row can share
    # a stamp; a report must never quietly overwrite the one before it.
    counter = 1

    while os.path.exists(path):
        path = f"{base}-{counter}.log"
        counter += 1

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(format_crash(exc_type, exc, tb))

    _rotate_logs()

    return path


def install_excepthook(on_crash: Callable[[str | None, str], None] | None = None) -> None:
    """Route uncaught exceptions through a log file and, optionally, *on_crash*.

    ``on_crash`` is called with two positional arguments: the path of the log
    that was just written (``None`` if even that failed) and the same report
    as text, so a dialog can show it and copy it without reading the file back.

    The previously installed hook is still called, so the traceback continues
    to reach stderr and the console stays as informative as it was.  Every
    stage is guarded independently: an exception raised while reporting a
    crash must not replace the crash being reported, and installing twice must
    not chain the hook to itself.
    """
    previous = getattr(sys.excepthook, "_dotgen_previous", sys.excepthook)

    def hook(exc_type, exc, tb) -> None:
        path: str | None = None
        details = ""

        try:
            details = format_crash(exc_type, exc, tb)
        except Exception:
            pass

        try:
            path = write_crash_log(exc_type, exc, tb)
        except Exception:
            path = None

        # stderr first: whatever the callback does -- a modal dialog, most
        # likely -- the console already holds the traceback by then.
        try:
            (previous or sys.__excepthook__)(exc_type, exc, tb)
        except Exception:
            pass

        if on_crash is not None:
            try:
                on_crash(path, details)
            except Exception:
                pass

    hook._dotgen_previous = previous  # type: ignore[attr-defined]
    sys.excepthook = hook


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _dotgen_version() -> str:
    try:
        import dotgen

        return str(getattr(dotgen, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _rotate_logs() -> None:
    """Keep the newest :data:`MAX_LOGS` crash logs and delete the rest."""
    try:
        entries = [
            os.path.join(log_dir(), name)
            for name in os.listdir(log_dir())
            if name.startswith("crash-") and name.endswith(".log")
        ]
    except OSError:
        return

    if len(entries) <= MAX_LOGS:
        return

    # Sort by name, not mtime: the stamp in the name is the truth about when
    # the crash happened, and a copied or restored file keeps it.
    for path in sorted(entries)[: len(entries) - MAX_LOGS]:
        try:
            os.remove(path)
        except OSError:
            pass


def _log_current_exception() -> None:
    """Record the exception being handled, without ever raising in turn."""
    try:
        write_crash_log(*sys.exc_info())
    except Exception:
        pass
