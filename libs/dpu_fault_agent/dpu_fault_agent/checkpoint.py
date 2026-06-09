from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_CHECKPOINT_PATH = ".dpu_fault_agent/checkpoints.sqlite"


@contextmanager
def sqlite_checkpointer(path: str = DEFAULT_CHECKPOINT_PATH) -> Iterator[SqliteSaver]:
    db_path = Path(path)
    if db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver
