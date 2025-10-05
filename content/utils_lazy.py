from __future__ import annotations

_cache = {"np": None, "pd": None, "plt": None}

def get_np():
    if _cache["np"] is None:
        import importlib
        _cache["np"] = importlib.import_module("numpy")
    return _cache["np"]

def get_pd():
    if _cache["pd"] is None:
        import importlib
        _cache["pd"] = importlib.import_module("pandas")
    return _cache["pd"]

def get_plt():
    if _cache["plt"] is None:
        import importlib
        _cache["plt"] = importlib.import_module("matplotlib.pyplot")
    return _cache["plt"]
