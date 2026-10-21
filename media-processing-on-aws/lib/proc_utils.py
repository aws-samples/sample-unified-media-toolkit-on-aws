"""
Shared subprocess runner for ffmpeg, ffprobe, and mxf2raw.
"""
from __future__ import annotations

import logging
import subprocess
import time

log = logging.getLogger(__name__)


def run(cmd: list, timeout: int = 3600, return_stderr: bool = False) -> str:
    """
    Run cmd, return stdout. Raises RuntimeError on non-zero exit.
    stderr is logged on failure. return_stderr=True returns stderr
    instead — for tools that report on stderr by design (ffmpeg's
    loudnorm print_format=json).
    """
    log.info("$ %s", " ".join(str(c) for c in cmd))
    t0 = time.monotonic()
    result = subprocess.run(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        [str(c) for c in cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = round(time.monotonic() - t0, 2)
    if result.returncode != 0:
        log.error("stderr:\n%s", result.stderr[-4000:])
        raise RuntimeError(
            f"Command exited {result.returncode} after {elapsed}s:\n"
            f"{result.stderr[-2000:]}"
        )
    log.info("Completed in %.2fs", elapsed)
    return result.stderr if return_stderr else result.stdout
