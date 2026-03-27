"""
pid_lock.py — 防止 launchd 重复启动同一脚本
用法：在脚本 if __name__ == "__main__" 开头加：
    from scripts.pid_lock import acquire_lock
    acquire_lock()
"""
import os
import sys
import atexit
from pathlib import Path

_LOCK_DIR = Path("/tmp")
_lock_path = None


def acquire_lock(name: str = None) -> None:
    """尝试获取进程锁。若同名进程已在运行则直接退出。"""
    global _lock_path

    script_name = name or Path(sys.argv[0]).stem
    _lock_path = _LOCK_DIR / f"moonx_{script_name}.pid"

    if _lock_path.exists():
        try:
            existing_pid = int(_lock_path.read_text().strip())
            # 检查进程是否真的还在运行
            os.kill(existing_pid, 0)
            print(f"[pid_lock] {script_name} already running (PID {existing_pid}), exiting.")
            sys.exit(0)
        except (ProcessLookupError, ValueError):
            # 进程已不存在，旧锁文件是残留，继续
            pass

    _lock_path.write_text(str(os.getpid()))
    atexit.register(_release_lock)


def _release_lock():
    if _lock_path and _lock_path.exists():
        _lock_path.unlink(missing_ok=True)
