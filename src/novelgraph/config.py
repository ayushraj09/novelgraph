"""
Zero-config resolution of the graph engine's storage roots.

Cognee requires `SYSTEM_ROOT_DIRECTORY` / `DATA_ROOT_DIRECTORY` to be
ABSOLUTE paths - relative paths raise an error - and if they're left
unset entirely, Cognee falls back to writing inside the virtualenv's
site-packages folder, which is easy to lose track of and gets wiped if
the venv is ever recreated.

Hand-editing an absolute path into `.env` for every machine/clone is
exactly the kind of setup friction this module removes: it resolves both
variables to absolute paths anchored to THIS project's own root (wherever
it was actually cloned) and applies them via `os.environ` before Cognee
is ever imported - so a fresh clone works with zero edits.

Precedence:
    1. An explicit value already in the environment / `.env` wins, as-is
       (just expanded/made absolute, e.g. `~` is resolved).
    2. Otherwise, default to `<project root>/.cognee_system` and
       `<project root>/.cognee_data`, created if they don't exist yet.

This module is imported as a side effect of importing the `novelgraph`
package itself (see `__init__.py`), so anything that does
`from novelgraph...import ...` gets these variables resolved
automatically, before any submodule's own `import cognee` runs. The one
place this can't help is a script that does `import cognee` before
importing anything from `novelgraph` - see `app.py`'s import order for
why that matters.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# src/novelgraph/config.py -> novelgraph -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_STORAGE_ROOTS = {
    "SYSTEM_ROOT_DIRECTORY": PROJECT_ROOT / ".cognee_system",
    "DATA_ROOT_DIRECTORY": PROJECT_ROOT / ".cognee_data",
}


def _resolve_storage_roots() -> None:
    for env_var, default_path in _DEFAULT_STORAGE_ROOTS.items():
        raw_value = os.environ.get(env_var, "").strip()
        if raw_value:
            # Respect an explicit override, but still make it absolute
            # (Cognee rejects relative paths, and this also expands `~`).
            os.environ[env_var] = str(Path(raw_value).expanduser().resolve())
            continue

        default_path.mkdir(parents=True, exist_ok=True)
        os.environ[env_var] = str(default_path)


_resolve_storage_roots()
