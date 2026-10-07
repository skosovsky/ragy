#!/usr/bin/env python3
"""AAA actual descendant-survival regression for both public command runners."""
import contextlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import check_release_consumer
import verify


class ProcessGroupTimeout(unittest.TestCase):
    def test_both_runners_kill_term_ignoring_descendant_after_leader_exit(self):
        for profile in ("verification", "consumer"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                # Arrange: child ignores TERM, closes output pipes and records readiness.
                root = Path(directory)
                pidfile = root / "child.pid"
                heartbeat = root / "child.heartbeat"
                child = ("import os,signal,time; from pathlib import Path; "
                         "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                         f"pulse=Path({str(heartbeat)!r}); pulse.write_text('ready'); "
                         f"Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                         "\nwhile True: pulse.write_text(str(time.monotonic_ns())); time.sleep(0.02)\n")
                leader = ("import subprocess,sys,time; from pathlib import Path; "
                          f"subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                          f"p=Path({str(pidfile)!r}); "
                          "\nwhile not p.exists(): time.sleep(0.01)\ntime.sleep(30)\n")
                command = [sys.executable, "-c", leader]
                try:
                    # Act: leader exits on TERM before its ignoring child.
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(subprocess.TimeoutExpired):
                            if profile == "verification":
                                verify.run(command, root, timeout=1)
                            else:
                                check_release_consumer.command(root, *command, timeout=1)
                    # Assert: no descendant work survives the timeout return.
                    self.assertTrue(pidfile.exists(), "child must be ready before timeout")
                    # A heartbeat proves active descendant work without requiring
                    # restricted platform process-table inspection; killed zombies
                    # awaiting init reaping cannot continue the pulse.
                    before = heartbeat.stat().st_mtime_ns
                    time.sleep(0.3)
                    self.assertEqual(heartbeat.stat().st_mtime_ns, before, "descendant still doing work")
                finally:
                    # A broken implementation must not leave this probe alive.
                    if pidfile.exists():
                        try:
                            os.kill(int(pidfile.read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
