"""Cooperative Slurm requeue signal shared by sequential and parallel runs."""

import signal
import threading


_REQUESTED = threading.Event()


def install():
  if hasattr(signal, 'SIGUSR1'):
    signal.signal(signal.SIGUSR1, lambda signum, frame: _REQUESTED.set())


def requested():
  return _REQUESTED.is_set()
