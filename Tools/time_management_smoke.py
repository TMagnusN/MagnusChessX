#!/usr/bin/env python3
"""Runtime smoke checks for MagnusChessX UCI time-control behavior."""

from __future__ import annotations

import argparse
import queue
import re
import subprocess
import threading
import time
from pathlib import Path


BESTMOVE_RE = re.compile(r"^bestmove ([a-h][1-8][a-h][1-8][qrbn]?|0000)(?: |$)")


class UciEngine:
    def __init__(self, executable: Path) -> None:
        self.process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to open engine pipes")
        self.lines: queue.Queue[str] = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\r\n"))

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def wait_for(self, prefix: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {prefix!r}")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(f"timed out waiting for {prefix!r}") from error
            if line.startswith(prefix):
                return line

    def ready(self) -> None:
        self.send("isready")
        self.wait_for("readyok", 10.0)

    def set_option(self, name: str, value: int | str) -> None:
        self.send(f"setoption name {name} value {value}")
        self.ready()

    def search(self, command: str, timeout: float) -> tuple[float, str]:
        started = time.monotonic()
        self.send(command)
        line = self.wait_for("bestmove ", timeout)
        elapsed = time.monotonic() - started
        if BESTMOVE_RE.match(line) is None:
            raise AssertionError(f"invalid bestmove line: {line}")
        return elapsed, line

    def assert_search_stays_active(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                return
            if line.startswith("bestmove "):
                raise AssertionError(f"search ended before control command: {line}")

    def close(self) -> None:
        if self.process.poll() is None:
            self.send("quit")
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)


def require_at_most(label: str, elapsed: float, maximum: float) -> None:
    if elapsed > maximum:
        raise AssertionError(f"{label}: {elapsed:.3f}s exceeds {maximum:.3f}s")


def run(executable: Path) -> None:
    engine = UciEngine(executable)
    try:
        engine.send("uci")
        engine.wait_for("uciok", 10.0)
        engine.ready()

        for threads in (1, 2, 8):
            engine.set_option("Threads", threads)
            engine.set_option("MultiPV", 1)
            engine.send("ucinewgame")
            engine.send("position startpos")
            elapsed, _ = engine.search("go movetime 100", 2.0)
            require_at_most(f"movetime/threads={threads}", elapsed, 0.25)

            engine.send("position startpos")
            elapsed, _ = engine.search("go wtime 1000 btime 1000", 2.0)
            require_at_most(f"clock/threads={threads}", elapsed, 1.05)

        engine.set_option("Threads", 1)
        engine.send("position startpos")
        engine.search("go depth 4", 5.0)
        engine.send("position startpos")
        engine.search("go nodes 5000", 5.0)

        engine.set_option("MultiPV", 2)
        engine.send("position startpos")
        elapsed, _ = engine.search("go movetime 100", 2.0)
        require_at_most("multipv movetime", elapsed, 0.25)
        engine.set_option("MultiPV", 1)

        engine.send("position startpos")
        engine.send("go infinite")
        engine.assert_search_stays_active(0.05)
        stopped = time.monotonic()
        engine.send("stop")
        engine.wait_for("bestmove ", 1.0)
        require_at_most("infinite stop", time.monotonic() - stopped, 0.50)

        engine.send("position startpos")
        engine.send("go ponder wtime 60000 btime 60000")
        engine.assert_search_stays_active(2.5)
        hit = time.monotonic()
        engine.send("ponderhit")
        engine.wait_for("bestmove ", 1.0)
        require_at_most("mature ponderhit", time.monotonic() - hit, 0.20)

        engine.send("position startpos")
        ponder_started = time.monotonic()
        engine.send("go ponder wtime 60000 btime 60000")
        engine.assert_search_stays_active(0.05)
        hit = time.monotonic()
        engine.send("ponderhit")
        engine.wait_for("bestmove ", 3.0)
        hit_elapsed = time.monotonic() - hit
        total_elapsed = time.monotonic() - ponder_started
        if hit_elapsed < 0.03:
            raise AssertionError("early ponderhit did not use its remaining budget")
        require_at_most("early ponder total", total_elapsed, 2.5)

        engine.send("ucinewgame")
        engine.ready()
        print("time management UCI smoke checks ok")
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    executable = args.engine.resolve()
    if not executable.is_file():
        parser.error(f"engine not found: {executable}")
    run(executable)


if __name__ == "__main__":
    main()
