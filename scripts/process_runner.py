#!/usr/bin/env python3
"""Bounded Unix subprocess groups for verification, including surviving descendants."""
import os
import signal
import subprocess


def signal_group(process, signal_number):
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass


def run_process(command, directory, env, timeout, capture=False, input_text=None):
    process = subprocess.Popen(command, cwd=directory, env=env, text=True, start_new_session=True,
                               stdin=subprocess.PIPE if input_text is not None else None,
                               stdout=subprocess.PIPE if capture else None,
                               stderr=subprocess.PIPE if capture else None)
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        signal_group(process, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A terminated leader does not prove the group is empty. A descendant
            # may ignore TERM or close inherited streams, so always kill the group.
            signal_group(process, signal.SIGKILL)
            process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
