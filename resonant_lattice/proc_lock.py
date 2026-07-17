"""proc_lock.py - cross-process finalize serialization for one memory DB.

The consolidation epoch and the dream cycle are each guarded in-process by a
threading lock, but nothing coordinates ACROSS processes: a long-lived gateway
and concurrent `hermes -z` overnight workers all open the same profile DB from
separate processes. Concurrent dream cycles from two processes can interleave
Hebbian maintenance (double decay, racing promotions/prunes, duplicate conflict
groups) - the same multi-process bug class as the cycle-counter clock rollback
fixed in store.increment_cycle().

FinalizeLock is an advisory file lock (`<db>.finalize.lock`, flock on POSIX,
msvcrt.locking on Windows) taken for the DURATION of one finalize unit (one
consolidation epoch, or one dream cycle). Scheduled mid-session work tries the
lock non-blocking and defers when another process is finalizing (episodes are
durable - they distill on the next trigger, mirroring the in-process skip
semantics). Forced session-end work waits its turn.

Locks are per-open-file-description, so two FinalizeLock instances contend
even inside one process - callers construct a fresh instance per finalize unit
and never share one across threads.

FAIL-OPEN: if the lock FILE cannot even be created (exotic permissions or a
read-only filesystem), acquire() warns once and proceeds unlocked - the
pre-existing behavior. Memory that consolidates unprotected beats memory that
silently stops consolidating.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

# How long a FORCED (session-end) finalize waits for other processes' finalize
# tails before giving up. Under N-way overnight concurrency the queue ahead is
# at most N-1 consolidation tails (minutes each on a cloud reason model).
FINALIZE_LOCK_WAIT = 1800.0   # seconds
_POLL = 0.5


class FinalizeLock:
    def __init__(self, db_path):
        self._path = str(db_path) + ".finalize.lock"
        self._fd = None
        self._unlocked_fallback = False

    def acquire(self, timeout: float = 0.0) -> bool:
        """Try to take the lock. timeout=0 is a single non-blocking attempt;
        otherwise poll until acquired or the deadline passes."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            got = self._try_once()
            if got is not False:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_POLL)

    def _try_once(self):
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_RDWR)
        except OSError as e:
            # Can't even create the lock file - fail OPEN (see module docstring).
            logger.warning(
                "FinalizeLock unavailable (%s) - proceeding without cross-process "
                "serialization for this finalize.", e)
            self._unlocked_fallback = True
            return None
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False   # held by another finalize - normal contention
        self._fd = fd
        return True

    def release(self) -> None:
        fd, self._fd = self._fd, None
        self._unlocked_fallback = False
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass
