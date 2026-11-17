"""Drive text mode and interrupt a long reply to verify barge-in plumbing."""
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = subprocess.Popen(
        [sys.executable, str(ROOT / "desktop" / "voice_call.py"), "--text", "--no-play", "--verbose"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    ready = threading.Event()
    out: list[str] = []

    def reader():
        for line in p.stdout:
            out.append(line)
            print("APP|", line, end="")
            if "等待输入" in line:
                ready.set()

    threading.Thread(target=reader, daemon=True).start()
    if not ready.wait(120):
        print("app not ready")
        p.kill()
        return 1

    time.sleep(0.5)
    p.stdin.write("请给我讲一个两百字左右的故事。\n")
    p.stdin.flush()
    print(">> sent long request")
    time.sleep(3.5)
    p.stdin.write("\n")  # empty line = barge-in
    p.stdin.flush()
    print(">> sent barge-in")
    time.sleep(1.0)
    p.stdin.write("给我讲个笑话吧。\n")
    p.stdin.flush()
    print(">> sent follow-up")
    time.sleep(9)
    p.stdin.write("/quit\n")
    p.stdin.flush()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    text = "".join(out)
    print("\nRESULT: interrupted =", "[打断]" in text)
    return 0 if "[打断]" in text else 2


if __name__ == "__main__":
    raise SystemExit(main())
