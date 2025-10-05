# content/feature_app.py — lazy-imports so NumPy/Pandas/Matplotlib load after the kernel is ready
from __future__ import annotations
import io
import ipywidgets as W  # pure-Python, safe to import at module level

# We'll attach these globals after lazy-importing
np = None
pd = None
plt = None

def _lazy_imports():
    """Import heavy libs only when needed (after kernel has loaded them)."""
    global np, pd, plt
    if np is not None and pd is not None and plt is not None:
        return
    import importlib
    np  = importlib.import_module("numpy")
    pd  = importlib.import_module("pandas")
    plt = importlib.import_module("matplotlib.pyplot")

def _load_meta() -> tuple["pd.DataFrame", str]:
    _lazy_imports()
    for p in ("data/meta.csv","content/data/meta.csv","/files/data/meta.csv","/drive/data/meta.csv"):
        try:
            return pd.read_csv(p), p
        except Exception:
            pass
    raise RuntimeError("meta.csv not found in expected locations.")

def _unit_col(df) -> str | None:
    return next((c for c in ("unit","unit_id","Unit","id","neuron_id") if c in df.columns), None)

def _numeric_cols(df, exclude: str | None) -> list[str]:
    _lazy_imports()
    cols = []
    for c in df.columns:
        if c == exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.95:
            df[c] = s
            cols.append(c)
    return cols

def _detect_features(df_in):
    _lazy_imports()
    df = df_in.copy()
    unit = _unit_col(df)
    preferred = [
        ["feature1","feature2","feature3"],
        ["Feature1","Feature2","Feature3"],
        ["feature_1","feature_2","feature_3"],
        ["feat1","feat2","feat3"],
        ["F1","F2","F3"],
    ]
    for cols in preferred:
        if all(c in df.columns for c in cols):
            for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")
            return df, unit, cols, [f"Using preferred columns: {', '.join(cols)}"]

    nums = _numeric_cols(df, unit)
    notes: list[str] = []
    if len(nums) >= 3:
        return df, unit, nums[:3], ["Using first three numeric columns."]
    if len(nums) == 2:
        a, b = nums
        f3 = f"feature3_mean({a},{b})"
        df[f3] = (df[a] + df[b]) / 2
        return df, unit, [a, b, f3], [f"Added {f3}."]
    if len(nums) == 1:
        a = nums[0]; s = df[a]
        std = float(s.std(ddof=0)) or 1.0
        rng = float(s.max() - s.min()) or 1.0
        f2 = f"feature2_z({a})"; f3 = f"feature3_minmax({a})"
        df[f2] = (s - float(s.mean())) / std
        df[f3] = (s - float(s.min())) / rng
        return df, unit, [a, f2, f3], [f"Added {f2} and {f3}."]

    n = len(df)
    df["feature1_index"] = np.arange(n)
    df["feature2_sqrt"]  = np.sqrt(np.arange(n))
    df["feature3_sin"]   = np.sin(np.arange(n))
    return df, unit, ["feature1_index","feature2_sqrt","feature3_sin"], ["No numeric columns; generated features."]

def _fig_png(df, cols: list[str], feature_idx: int, unit: str | None) -> bytes:
    _lazy_imports()
    idx = int(feature_idx) - 1
    if not (0 <= idx < len(cols)):
        raise IndexError(f"feature index {feature_idx} out of range 1..{len(cols)}")
    col = cols[idx]
    y = df[col].to_numpy()
    x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, y, marker="o", linestyle="-")
    ax.set_title(f"{col} across all units (n={len(y)})")
    ax.set_xlabel(unit if unit else "Unit index")
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def build_app():
    _lazy_imports()
    df_raw, path = _load_meta()
    df, unit, cols, notes = _detect_features(df_raw)
    info = W.HTML(
        f"<b>Loaded:</b> <code>{path}</code><br>"
        f"<b>Features:</b> {', '.join(cols)}<br>"
        f"<small>{' '.join(notes)}</small>"
    )
    slider = W.IntSlider(value=1, min=1, max=3, step=1, description="Feature", continuous_update=False)
    img = W.Image(format="png")
    def redraw(_=None): img.value = _fig_png(df, cols, slider.value, unit)
    slider.observe(redraw, names="value")
    redraw()
    return W.VBox([info, slider, img], layout=W.Layout(width="900px"))

def main():
    return build_app()
