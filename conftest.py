"""Root conftest so `detection` (a plain namespace package, no __init__.py)
is importable from tests/ as `from detection import features`, the same way
`detection/main.py` imports it when run as `uvicorn detection.main:app` from
the repo root. Pytest adds this file's directory to sys.path when it
collects conftest.py files, so this needs no code - just to exist here.
"""
