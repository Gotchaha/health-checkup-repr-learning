# DL_profiling/multiprocessing_utils.py
"""
Tiny utilities for DataLoader workers: per-worker init/exit mem logging.

Design:
- One log file per worker process: <log_dir>/dl_worker_<PID>.log
- We write one "init" line at startup and one "exit" line at atexit.
- Peak memory uses ru_maxrss (platform-aware scaling); PyArrow bytes is optional.
- The log dir is taken from env DL_WORKER_LOG_DIR; falls back to tempfile.gettempdir().
"""

import os
import sys
import atexit
import signal
import logging
import resource
import platform

def _ru_maxrss_mb() -> float:
    """Return peak RSS in MB (platform-aware)."""
    # Linux: KB; macOS: bytes
    scale = 1/1024.0 if platform.system() == "Linux" else 1/(1024.0 * 1024.0)
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    except Exception:
        return -1.0

def _arrow_mb() -> float:
    """Return PyArrow total allocated bytes in MB (or -1 if unavailable)."""
    try:
        import pyarrow as pa
        return pa.total_allocated_bytes() / (1024.0 ** 2)
    except Exception:
        return -1.0


def worker_init_with_memlog(worker_id: int) -> None:
    # Ensure a console handler in the worker (goes to stderr by default)
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root.addHandler(h)
        root.setLevel(logging.INFO)
    log = logging.getLogger("dl.worker")

    # Ignore Ctrl-C in workers; main process handles it
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    # Print INIT line
    try:
        log.info(f"[worker {os.getpid()} init] id={worker_id} | RSS={_ru_maxrss_mb():.1f} MB | Arrow={_arrow_mb():.1f} MB")
        for h in log.handlers:
            try: h.flush()
            except Exception: pass
        sys.stderr.flush()
    except Exception:
        pass

    # Common exit emitter (used by atexit and SIGTERM)
    def _emit_exit(tag: str = "exit") -> None:
        try:
            log.info(f"[worker {os.getpid()} {tag}] id={worker_id} | peakRSS={_ru_maxrss_mb():.1f} MB | Arrow={_arrow_mb():.1f} MB")
            for h in log.handlers:
                try: h.flush()
                except Exception: pass
            sys.stderr.flush()
        except Exception:
            pass

    # atexit: covers clean/normal interpreter shutdown
    atexit.register(_emit_exit, "exit")

    # SIGTERM: what DataLoader typically sends to stop workers
    try:
        def _on_sigterm(signum, frame):
            _emit_exit("exit")   # print and flush
            os._exit(0)          # immediate exit (we've already logged)
        signal.signal(signal.SIGTERM, _on_sigterm)
    except Exception:
        pass

