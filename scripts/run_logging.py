#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行日志、进度条、耗时与金额格式化工具。"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

USD_TO_CNY = 7.2


def format_duration(seconds) -> str:
    seconds = max(0, int(round(seconds or 0)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def format_money(usd) -> str:
    if usd is None:
        return "n/a"
    return f"${usd:.4f} USD / {usd * USD_TO_CNY:.4f} CNY"


def format_bar(done: int, total: int, width: int = 28) -> str:
    total = max(int(total or 0), 1)
    ratio = min(max((done or 0) / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class RunLog:
    """线程安全的运行日志文件。"""

    def __init__(self, name: str, log_dir: str = "", echo: bool = False):
        base = Path(log_dir) if log_dir else Path(__file__).resolve().parent.parent / "logs"
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = base / f"{name}-{stamp}-{os.getpid()}.log"
        self.echo = echo
        self._lock = threading.Lock()
        self._fh = self.path.open("w", encoding="utf-8")
        self.write(f"===== {name} start {datetime.now().isoformat(timespec='seconds')} =====")

    def write(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(line + "\n")
            self._fh.flush()
        if self.echo:
            print(line, flush=True)

    def close(self) -> None:
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===== done =====\n")
            self._fh.close()


class ProgressReporter:
    """在控制台输出进度条，并把逐篇明细写入日志。"""

    def __init__(
        self,
        total: int,
        label: str = "进度",
        log: RunLog = None,
        cost_fn=None,
        stream=None,
        width: int = 28,
    ):
        self.total = max(int(total or 0), 0)
        self.label = label
        self.log = log
        self.cost_fn = cost_fn
        self.stream = stream or sys.stdout
        self.width = width
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.lock = threading.Lock()
        self.done = 0
        self.counts = {"ok": 0, "skip": 0, "error": 0, "other": 0}
        self.started_at = time.monotonic()
        self._last_line_len = 0

    def start(self) -> None:
        message = f"[Start] {self.label}: 共 {self.total} 篇，开始计时 ..."
        print(message, flush=True)
        if self.log:
            self.log.write(message)

    def _status_key(self, status: str) -> str:
        status = (status or "").lower()
        if status in {"ok", "success", "done"}:
            return "ok"
        if status in {"skip", "skipped"}:
            return "skip"
        if status in {"error", "failed", "fail"}:
            return "error"
        return "other"

    def _cost(self):
        if not self.cost_fn:
            return None
        try:
            return self.cost_fn()
        except Exception:
            return None

    def _line(self) -> str:
        elapsed = time.monotonic() - self.started_at
        rate = elapsed / self.done if self.done else 0.0
        eta = rate * max(self.total - self.done, 0)
        percent = (self.done / self.total * 100) if self.total else 100.0
        cost = self._cost()
        parts = [
            f"[{self.done}/{self.total}]",
            format_bar(self.done, self.total, self.width),
            f"{percent:5.1f}%",
            f"OK {self.counts['ok']}",
            f"Skip {self.counts['skip']}",
            f"Err {self.counts['error']}",
            f"耗时 {format_duration(elapsed)}",
            f"ETA {format_duration(eta) if self.done < self.total else '00s'}",
        ]
        if cost is not None:
            parts.append(format_money(cost))
        return " | ".join(parts)

    def _render(self, line: str) -> None:
        if self.interactive:
            padded = line.ljust(self._last_line_len)
            self.stream.write("\r" + padded)
            self.stream.flush()
            self._last_line_len = len(line)
        else:
            print(line, flush=True)

    def update(self, status: str, title: str = "", detail: str = "") -> None:
        with self.lock:
            self.done += 1
            key = self._status_key(status)
            self.counts[key] += 1
            line = self._line()
            self._render(line)
            if self.log:
                safe_detail = reflow(detail or title)
                self.log.write(f"{line} | {status or '-'} {safe_detail}".strip())
            if key == "error" and detail:
                if self.interactive:
                    self.stream.write("\n")
                print(reflow(detail), flush=True)

    def finish(self) -> None:
        with self.lock:
            line = self._line()
            if self.interactive:
                self.stream.write("\r" + line.ljust(self._last_line_len) + "\n")
                self.stream.flush()
            elif self.done < self.total:
                print(line, flush=True)
            if self.log:
                self.log.write(
                    f"[Done] {self.label}: OK={self.counts['ok']} "
                    f"Skip={self.counts['skip']} Err={self.counts['error']} "
                    f"Other={self.counts['other']} elapsed={format_duration(time.monotonic() - self.started_at)}"
                )


def reflow(text: str) -> str:
    return " ".join((text or "").split())
