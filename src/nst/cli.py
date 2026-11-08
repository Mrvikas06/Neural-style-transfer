"""Command-line interface (CLI) entry point for Neural Style Transfer.

Provides the canonical console script mapping for 'nst' and 'python -m nst.cli'.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main() -> None:
    """Execute the headless NST CLI engine."""
    from scripts.run_nst import main as runner_main

    runner_main()


if __name__ == "__main__":
    main()
