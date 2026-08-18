"""Guard against accidental use of the official 20,509-row held-out test set.

Round-2 model search rule (claude_code_round2_pe_rankformer_model_search.md §2, §36):
the held-out set was already used for the round-1 final evaluation and must not be
touched again until a round-2 model/ensemble is explicitly frozen. Every script that
can evaluate on it must require an explicit `--allow-heldout-evaluation` flag and log
the access.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import json
import sys
from pathlib import Path

LOG_PATH = Path("logs/heldout_evaluations.log")


def require_heldout_permission(allowed: bool, *, script: str, reason: str, n_rows: int) -> None:
    """Abort unless `allowed` is True; otherwise append a timestamped audit record.

    `allowed` should be the value of an `--allow-heldout-evaluation` CLI flag: callers
    must not default it to True.
    """
    if not allowed:
        sys.exit(
            f"REFUSING to evaluate on the official held-out test set from {script!r}.\n"
            "This set was already used for the round-1 final evaluation and is locked "
            "during round-2 model search (see claude_code_round2_pe_rankformer_model_search.md "
            "§2). Pass --allow-heldout-evaluation only after the round-2 model is frozen."
        )

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "script": script,
        "reason": reason,
        "n_rows": n_rows,
        "argv": sys.argv,
        "user": getpass.getuser(),
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[heldout-guard] LOGGED held-out evaluation -> {LOG_PATH}", file=sys.stderr)
