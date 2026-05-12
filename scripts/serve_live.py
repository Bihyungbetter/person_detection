from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dementia_tracker.cli import main


if __name__ == "__main__":
    if "--log" in sys.argv:
        log_index = sys.argv.index("--log")
        log_path = ROOT / sys.argv[log_index + 1]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file
        del sys.argv[log_index : log_index + 2]
    main(["serve-live", *sys.argv[1:]])
