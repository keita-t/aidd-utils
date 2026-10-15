#!/usr/bin/env python3
"""Claude Code のプロンプトキャッシュ維持 hook。

バックグラウンドタスク（バックグラウンド実行の Bash と Agent）の完了を待ってセッションが
アイドルになっている間、最後の応答から一定時間（既定 50 分）が過ぎるたびに asyncRewake で
モデルを起こし、キャッシュ（TTL 1 時間）を読み直させて期限を延ばす。

サブコマンド:
  stop           Stop hook（asyncRewake）。未完了タスクがあれば待機タイマーを 1 本動かす
  session-start  SessionStart hook。開始時刻を記録し、それ以前に起動したタスクを対象外にする
  session-end    SessionEnd hook。動いている待機タイマーを止める

待機タイマーは次のどれかで、モデルを起こさずに終わる。
  - 未完了タスクが無くなった（<task-notification> がトランスクリプトに記録された）
  - 新しい Stop が来た（そのターンの終わりに新しいタイマーへ引き継ぐ）
  - SessionEnd が来た、または親プロセス（Claude Code）が終了した
  - 最後の応答から TTL を過ぎていた（スリープ復帰など。もう維持できないので起こさない）
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime

STATE_DIR = os.environ.get("CLAUDE_CACHE_KEEPALIVE_STATE_DIR") or os.path.join(
    os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir(), "claude-cache-keepalive"
)

NOTIFICATION_RE = re.compile(r"<task-notification>(.*?)</task-notification>", re.S)
TASK_ID_RE = re.compile(r"<task-id>([^<]+)</task-id>")
STATUS_RE = re.compile(r"<status>([^<]+)</status>")
# completed / failed / killed / stopped 以外の status は確認できていない。
# 未知の status は完了扱いにして、起こし続ける側に倒さない。
NON_TERMINAL_STATUSES = {"running", "pending"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["stop", "session-start", "session-end"])
    p.add_argument("--keepalive-min", type=float, default=50.0, help="最後の応答から起こすまでの分数")
    p.add_argument("--ttl-min", type=float, default=60.0, help="キャッシュ TTL。これを過ぎていたら起こさない")
    p.add_argument("--max-hours", type=float, default=12.0, help="これより前に起動したタスクは待たない")
    p.add_argument("--poll-sec", type=float, default=30.0, help="完了・中止を確認する間隔")
    return p.parse_args()


def state_path(session_id, suffix):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)
    return os.path.join(STATE_DIR, f"{safe}.{suffix}")


def read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def write_atomic(path, text):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def log(session_id, message):
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(state_path(session_id, "log"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} [{os.getpid()}] {message}\n")
    except OSError:
        pass


def parse_timestamp(record):
    try:
        return datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")).timestamp()
    except (KeyError, AttributeError, ValueError):
        return 0.0


def load_json(raw):
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def notification_text(record):
    """ハーネスが入れたタスク完了通知の本文を返す。ツール出力や応答文中の引用は対象外。"""
    kind = record.get("type")
    if kind == "queue-operation" and record.get("operation") == "enqueue":
        content = record.get("content")
    elif (record.get("origin") or {}).get("kind") == "task-notification":
        content = (record.get("message") or {}).get("content")
    else:
        return ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


class Transcript:
    """トランスクリプト（JSONL）を追記分だけ読み、タスクの起動・完了と最後の応答時刻を追う。"""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.launched = {}  # task id -> 起動時刻
        self.finished = set()
        self.last_assistant = 0.0

    def update(self, track_activity=True):
        try:
            if os.path.getsize(self.path) < self.offset:
                self.__init__(self.path)
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                for raw in f:
                    if not raw.endswith(b"\n"):
                        break  # 書きかけの行は次回読む
                    self.offset += len(raw)
                    self._read_line(raw, track_activity)
        except OSError:
            pass

    def _read_line(self, raw, track_activity):
        if b'"backgroundTaskId"' in raw or b'"isAsync"' in raw:
            record = load_json(raw)
            result = record.get("toolUseResult") if record else None
            if isinstance(result, dict):
                task_id = result.get("backgroundTaskId") or (result.get("isAsync") is True and result.get("agentId"))
                if isinstance(task_id, str):
                    self.launched.setdefault(task_id, parse_timestamp(record))
        if b"Successfully stopped task" in raw:  # TaskStop で止めたタスクには完了通知が来ない
            record = load_json(raw)
            result = record.get("toolUseResult") if record else None
            if isinstance(result, dict) and str(result.get("message", "")).startswith("Successfully stopped task"):
                task_id = result.get("task_id")
                if isinstance(task_id, str):
                    self.finished.add(task_id)
        if b"<task-notification>" in raw:
            record = load_json(raw)
            for block in NOTIFICATION_RE.findall(notification_text(record) if record else ""):
                task_id = TASK_ID_RE.search(block)
                status = STATUS_RE.search(block)
                if task_id and not (status and status.group(1).strip() in NON_TERMINAL_STATUSES):
                    self.finished.add(task_id.group(1).strip())
        if track_activity and b'"assistant"' in raw:
            record = load_json(raw)
            if record and record.get("type") == "assistant":
                self.last_assistant = max(self.last_assistant, parse_timestamp(record))

    def pending(self, since):
        return sorted(t for t, at in self.launched.items() if t not in self.finished and at >= since)


def keepalive_message(pending, idle_min):
    return (
        "[cache-keepalive] プロンプトキャッシュ維持のための自動入力です（エラーではありません）。"
        f"バックグラウンドタスク {', '.join(pending)} の完了待ちで、最後の応答から約 {idle_min:.0f} 分が経過しました。"
        "ツールを呼ばず、タスクの状況確認もせず、「待機中」とだけ返してこのターンを終了してください。\n"
    )


def run_stop(hook, args):
    # claude -p（streaming input なし）では asyncRewake が同期実行になり、終了を待たせてしまう。
    if os.environ.get("CLAUDE_CODE_ENTRYPOINT") == "sdk-cli":
        return 0
    session_id = hook.get("session_id") or ""
    transcript_path = hook.get("transcript_path") or ""
    if not session_id or not transcript_path:
        return 0

    timer_path = state_path(session_id, "timer")
    token = f"{os.getpid()}:{time.time_ns()}"
    write_atomic(timer_path, token)  # 前のターンのタイマーはこれを見て終わる

    started = time.time()
    session_started = float(read_text(state_path(session_id, "started")) or 0)
    since = max(session_started, started - args.max_hours * 3600)
    transcript = Transcript(transcript_path)
    transcript.update(track_activity=False)
    pending = transcript.pending(since)
    if not pending:
        return 0

    log(session_id, f"timer start: pending={pending}")
    parent = os.getppid()
    last_activity = started  # Stop はターン最後の応答の直後に呼ばれる
    keepalive_sec = args.keepalive_min * 60
    while True:
        remaining = last_activity + keepalive_sec - time.time()
        if remaining <= 0:
            break
        time.sleep(min(args.poll_sec, remaining))
        if read_text(timer_path) != token:
            log(session_id, "timer end: superseded or session ended")
            return 0
        if os.getppid() != parent:
            log(session_id, "timer end: parent exited")
            return 0
        transcript.update()
        last_activity = max(last_activity, transcript.last_assistant)
        pending = transcript.pending(since)
        if not pending:
            log(session_id, "timer end: background tasks finished")
            return 0

    idle_sec = time.time() - last_activity
    if idle_sec >= args.ttl_min * 60:
        log(session_id, f"timer end: idle {idle_sec / 60:.0f} min exceeds TTL, not waking")
        return 0
    log(session_id, f"wake: pending={pending}")
    sys.stderr.write(keepalive_message(pending, idle_sec / 60))
    return 2


def main():
    args = parse_args()
    try:
        hook = json.load(sys.stdin)
    except ValueError:
        return 0
    session_id = hook.get("session_id") or ""
    if args.command == "stop":
        return run_stop(hook, args)
    if not session_id:
        return 0
    if args.command == "session-start":
        if hook.get("source") != "compact":
            write_atomic(state_path(session_id, "started"), str(time.time()))
    elif args.command == "session-end":
        remove(state_path(session_id, "timer"))
        remove(state_path(session_id, "started"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
