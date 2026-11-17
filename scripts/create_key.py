"""Convenience wrapper: python scripts/create_key.py --name demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from native_voice.cli.main import main  # noqa: E402

if __name__ == "__main__":
    main(["key", "create", *sys.argv[1:]])
