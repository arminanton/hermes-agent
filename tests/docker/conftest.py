"""Shared fixtures for docker-image integration tests.

Tests in this directory build the image with the current ``Dockerfile``
and exercise it via ``docker run``. They skip when Docker is unavailable
(e.g. on developer laptops without a daemon).

Override the image with ``HERMES_TEST_IMAGE`` env var to point at a pre-built
image (faster local iteration); otherwise the ``built_image`` fixture builds
the repo's Dockerfile once per session.

Docker tests need longer timeouts than the suite default (30s), so every
test under this directory is granted a 180s default via
``pytest.mark.timeout`` applied at collection time.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator

import pytest

IMAGE_TAG = os.environ.get("HERMES_TEST_IMAGE", "hermes-agent-harness:latest")


def _docker_available() -> bool:
    """Return True iff a docker CLI is on PATH and the daemon answers."""
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


# ---------------------------------------------------------------------------
# Leaked-container reaper — make the fuse-overlayfs leak structurally impossible
# ---------------------------------------------------------------------------
#
# Rootless podman (the `docker` shim on this host) starts one
# ``fuse-overlayfs`` daemon per container to mount its overlay rootfs.
# That daemon is only torn down when the container is *removed*. Our
# per-test cleanup lives in Python fixture teardown (``docker rm -f`` in
# the ``container_name`` / ``restart_container`` fixtures), which is the
# right thing — until the pytest process is hard-killed (SIGKILL, hard
# timeout, OOM, a runaway ``-x`` abort). Teardown never runs, the
# container is never removed, and its fuse-overlayfs reparents to PID 1
# and pins the mount forever. Run the suite enough times with abnormal
# exits and you accumulate thousands of orphaned daemons.
#
# Python ``finally`` / ``yield`` teardown fundamentally cannot survive
# SIGKILL, so per-fixture cleanup can never fully close this. The fix is
# a sweep that does not depend on this process surviving:
#
#   * ``pytest_sessionstart`` removes any test containers left behind by
#     a *previous* run that died without cleanup. This is the part that
#     survives SIGKILL — the next session always cleans the last one's
#     corpses, so leaks can never accumulate across runs.
#   * ``pytest_sessionfinish`` / ``pytest_unconfigure`` sweep the current
#     run's containers on the way out for normal and most-abnormal exits.
#
# Every test container is named by one of exactly two fixtures, with a
# stable prefix (``hermes-test-`` or ``hermes-restart-``), so a
# name-prefix sweep catches 100% of them without touching a single
# launch site or interfering with the ``docker restart`` tests (which
# must NOT use ``--rm``). ``docker rm -f`` removes the container AND
# unmounts its overlay, which reaps the fuse-overlayfs daemon.

_TEST_CONTAINER_PREFIXES = ("hermes-test-", "hermes-restart-")
_TEST_VOLUME_PREFIX = "hermes-restart-vol-"


def _sweep_leaked_test_containers() -> None:
    """Remove any containers/volumes left over from test runs.

    Best-effort and never raises: a reaper that crashes the session is
    worse than a missed sweep. Matches by the stable name prefixes the
    fixtures use so it can run at session start (prior run's corpses) and
    session end (this run's) without knowing which tests ran.
    """
    if _docker(  # cheap daemon liveness probe; skip silently if down
        ["docker", "info"], capture_output=True, timeout=5,
    ) is None:
        return
    try:
        listed = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "name=hermes-test-",
             "--filter", "name=hermes-restart-"],
            capture_output=True, text=True, timeout=15,
        )
        ids = [cid for cid in listed.stdout.split() if cid.strip()]
        if ids:
            subprocess.run(
                ["docker", "rm", "-f", *ids],
                capture_output=True, timeout=60,
            )
    except (subprocess.TimeoutExpired, OSError):
        pass
    # Named volumes from the restart fixture leak the same way; sweep them
    # too. Volumes can't be force-removed while a container holds them, so
    # this runs after the container removal above.
    try:
        vols = subprocess.run(
            ["docker", "volume", "ls", "-q", "--filter",
             f"name={_TEST_VOLUME_PREFIX}"],
            capture_output=True, text=True, timeout=15,
        )
        names = [v for v in vols.stdout.split() if v.strip()]
        if names:
            subprocess.run(
                ["docker", "volume", "rm", "-f", *names],
                capture_output=True, timeout=30,
            )
    except (subprocess.TimeoutExpired, OSError):
        pass


def _docker(cmd, **kw):
    """Run a docker command, returning the CompletedProcess or None.

    Thin wrapper so the sweep can probe daemon liveness without raising.
    """
    try:
        return subprocess.run(cmd, **kw)
    except (subprocess.TimeoutExpired, OSError):
        return None


def pytest_sessionstart(session):  # noqa: D401 - pytest hook
    """Reap test containers orphaned by a previous (possibly killed) run.

    This is the SIGKILL-proof half of the leak fix: even if the last
    session was hard-killed before its own cleanup could run, this clears
    its leaked containers before we add more, so orphans can never pile
    up across runs.
    """
    _sweep_leaked_test_containers()


def pytest_sessionfinish(session, exitstatus):  # noqa: D401 - pytest hook
    """Sweep this run's test containers on the way out (normal exits)."""
    _sweep_leaked_test_containers()


def pytest_collection_modifyitems(config, items):  # noqa: D401 - pytest hook
    """Apply docker-suite policy: timeout bump + skip on missing docker."""
    docker_ok = _docker_available()
    skip_docker = pytest.mark.skip(
        reason="Docker not available or daemon not running",
    )
    extend_timeout = pytest.mark.timeout(180)
    for item in items:
        if "tests/docker/" not in str(item.fspath).replace(os.sep, "/"):
            continue
        item.add_marker(extend_timeout)
        if not docker_ok:
            item.add_marker(skip_docker)


@pytest.fixture(scope="session")
def built_image() -> str:
    """Build the image once per test session.

    Override with ``HERMES_TEST_IMAGE`` env var to point at a pre-built
    image (faster local iteration).
    """
    if os.environ.get("HERMES_TEST_IMAGE"):
        return IMAGE_TAG
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_TAG, repo_root],
        capture_output=True, text=True, timeout=1200,
    )
    assert result.returncode == 0, (
        f"docker build failed:\n{result.stderr[-2000:]}"
    )
    return IMAGE_TAG


@pytest.fixture
def container_name(request) -> Iterator[str]:
    """Generate a unique container name and ensure cleanup on test exit."""
    safe = request.node.name.replace("[", "_").replace("]", "_")
    name = f"hermes-test-{safe}"
    yield name
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, timeout=10,
    )


# ---------------------------------------------------------------------------
# docker_exec — default to the unprivileged hermes user
# ---------------------------------------------------------------------------
#
# Background: every Hermes runtime path inside the container drops to UID
# 10000 (the ``hermes`` user) via ``s6-setuidgid hermes``. ``docker exec``
# without ``-u`` runs as root, which is **not** representative of how
# production code executes. PR #30136 review caught a real regression
# this way — ``Path('/proc/1/exe').resolve()`` works as root and silently
# fails (PermissionError swallowed) for hermes, so a test that ran as root
# couldn't catch a feature that was inert for the actual runtime user.
#
# Tests in this directory MUST exercise the realistic user context. The
# helpers below run every probe under ``-u hermes`` unless a specific
# test explicitly opts into ``user="root"`` (rare — e.g. inspecting
# /proc/1/exe itself, chowning a volume).
# ---------------------------------------------------------------------------


def docker_exec(
    container: str,
    *args: str,
    user: str = "hermes",
    timeout: int = 30,
    extra_docker_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run a command inside ``container`` as ``user`` (default: hermes).

    Returns the CompletedProcess with text=True, capture_output=True.

    Pass ``user="root"`` only when the test specifically needs root
    capabilities (e.g. reading /proc/1/exe, manipulating ownership).
    Most tests should use the default.
    """
    cmd = ["docker", "exec", "-u", user, *extra_docker_args, container, *args]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )


def docker_exec_sh(
    container: str,
    command: str,
    *,
    user: str = "hermes",
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run ``sh -c <command>`` inside the container as ``user``."""
    return docker_exec(
        container, "sh", "-c", command, user=user, timeout=timeout,
    )
