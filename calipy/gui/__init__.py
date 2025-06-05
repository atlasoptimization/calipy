# calipy/gui/__init__.py
try:
    from .app import main, launch_designer           # re-export
except ModuleNotFoundError as exc:                   # PySide6 missing?
    _import_error = exc

def main():
    """Entry point added in pyproject.toml."""
    if '_import_error' in globals():
        raise RuntimeError(
            "GUI extras not installed.\n"
            "Run:  pip install calipy-ppl[gui]"
        ) from _import_error
    launch_designer()

