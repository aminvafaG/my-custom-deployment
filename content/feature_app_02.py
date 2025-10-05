# content/feature_app.py — JupyterLite-friendly dashboard (no-code view in notebook)
# Filters by numeric features and optional 'layer'; overlays population, mean, and a selected unit
# Plots orientation tuning + PSTH; shows selected unit's feature row.
# Works with: content/data/meta.csv, tuning_curves.npz, psth.npz

from __future__ import annotations
import io, math
import ipywidgets as W

# Lazy imports so importing this module never fails on cold kernels
np = None
pd = None
plt = None

def _lazy_imports():
    global np, pd, plt
    if np is not None:
        return
    import importlib
    np  = importlib.import_module("numpy")
    pd  = importlib.import_module("pandas")
    plt = importlib.import_module("matplotlib.pyplot")

# ---- data loading ----
_DATA_PATHS = (
    ("content/data/meta.csv", "content/data/tuning_curves.npz", "content/data/psth.npz"),
    ("data/meta.csv",         "data/tuning_curves.npz",         "data/psth.npz"),
    ("/files/data/meta.csv",  "/files/data/tuning_curves.npz",  "/files/data/psth.npz"),
    ("/drive/data/meta.csv",  "/drive/data/tuning_curves.npz",  "/drive/data/psth.npz"),
)

def _load_all():
    _lazy_imports()
    last_err = None
    for meta_p, tc_p, psth_p in _DATA_PATHS:
        try:
            df = pd.read_csv(meta_p)
            tc = np.load(tc_p, allow_pickle=True)
            psth = np.load(psth_p, allow_pickle=True)
            angles = np.asarray(tc["angles"])
            tc_curves = np.asarray(tc["curves"])
            t = np.asarray(psth["t"])
            psth_curves = np.asarray(psth["curves"])
            return (df, meta_p), (angles, tc_curves, tc_p), (t, psth_curves, psth_p)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load data from known paths. Last error: {last_err!r}")

def _unit_col(df) -> str:
    for c in ("unit_id","unit","id","neuron_id","Unit","Unit_id"):
        if c in df.columns: return c
    return ""

def _numeric_feature_columns(df, exclude_cols=()):
    _lazy_imports()
    numeric_cols = []
    for c in df.columns:
        if c in exclude_cols: 
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.95:
            df[c] = s
            numeric_cols.append(c)
    return numeric_cols

# ---- plotting ----
def _to_png(fig, dpi=144):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def _plot_tuning(angles, tc_curves, idxs, highlight_idx=None):
    _lazy_imports()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if len(idxs):
        for i in idxs:
            ax.plot(angles, tc_curves[i], lw=0.8, alpha=0.20)
        ax.plot(angles, tc_curves[idxs].mean(axis=0), lw=2.5)
    if highlight_idx is not None and 0 <= highlight_idx < tc_curves.shape[0]:
        ax.plot(angles, tc_curves[highlight_idx], lw=3.2)
    ax.set_title("Orientation tuning")
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Response (a.u.)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _to_png(fig)

def _plot_psth(t, psth_curves, idxs, highlight_idx=None):
    _lazy_imports()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if len(idxs):
        for i in idxs:
            ax.plot(t, psth_curves[i], lw=0.8, alpha=0.20)
        ax.plot(t, psth_curves[idxs].mean(axis=0), lw=2.5)
    if highlight_idx is not None and 0 <= highlight_idx < psth_curves.shape[0]:
        ax.plot(t, psth_curves[highlight_idx], lw=3.2)
    ax.set_title("PSTH")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (sp/s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _to_png(fig)

# ---- UI app ----
def build_app():
    _lazy_imports()
    (df, meta_p), (angles, tc_curves, tc_p), (t, psth_curves, psth_p) = _load_all()
    df = df.copy()
    N = len(df)

    unit_col = _unit_col(df)
    if unit_col == "":
        df["unit_id"] = np.arange(1, N+1)
        unit_col = "unit_id"

    # optional categorical 'layer'
    layer_col = None
    for c in ("layer","Layer","lamina","Lamina"):
        if c in df.columns:
            layer_col = c
            break

    num_cols = _numeric_feature_columns(df, exclude_cols=(unit_col, layer_col) if layer_col else (unit_col,))
    # sliders
    sliders = {}
    for c in num_cols:
        s = df[c].astype(float)
        lo = float(np.nanmin(s))
        hi = float(np.nanmax(s))
        step = (hi - lo)/200.0 if math.isfinite(hi-lo) and (hi-lo) > 0 else 0.01
        sliders[c] = W.FloatRangeSlider(
            value=[lo, hi], min=lo, max=hi, step=step,
            description=c, readout_format=".3f", continuous_update=False, layout=W.Layout(width="100%")
        )

    # layer filter
    layer_ms = None
    if layer_col:
        layers = sorted(x for x in df[layer_col].dropna().unique().tolist())
        layer_ms = W.SelectMultiple(options=layers, value=tuple(layers), rows=min(6, len(layers)),
                                    description="layer")

    # unit dropdown (populated on filter)
    unit_dd = W.Dropdown(options=[("— none —", -1)], value=-1, description="unit")

    # outputs
    where_loaded = W.HTML(
        f"<small><b>Loaded</b>: <code>{meta_p}</code>, <code>{tc_p}</code>, <code>{psth_p}</code></small>"
    )
    status = W.HTML()
    tuning_img = W.Image(format="png", layout=W.Layout(width="100%"))
    psth_img   = W.Image(format="png", layout=W.Layout(width="100%"))
    features_html = W.HTML()

    reset_btn = W.Button(description="Reset filters", icon="undo")

    ctrls = []
    if layer_ms: ctrls.append(layer_ms)
    ctrls.extend(sliders.values())
    controls = W.VBox([
        W.HTML("<b>Filters</b>"),
        *ctrls,
        reset_btn,
        W.HTML("<hr>"),
        unit_dd,
    ], layout=W.Layout(width="320px", overflow_y="auto", max_height="80vh"))

    tabs = W.Tab(children=[tuning_img, psth_img, features_html])
    tabs.set_title(0, "Tuning")
    tabs.set_title(1, "PSTH")
    tabs.set_title(2, "Selected unit features")

    app = W.AppLayout(
        header=where_loaded,
        left_sidebar=controls,
        center=tabs,
        right_sidebar=None,
        footer=status,
        # NOTE: Use explicit px/fr values; 'auto' is not allowed here in Lite
        pane_widths=["340px", "1fr", "0px"],
        pane_heights=["30px", "1fr", "30px"],
    )

    def _current_mask():
        m = np.ones(N, dtype=bool)
        if layer_ms:
            sel = set(layer_ms.value)
            if sel:
                m &= df[layer_col].isin(sel).to_numpy()
        for c, sl in sliders.items():
            lo, hi = sl.value
            s = df[c].to_numpy(dtype=float)
            m &= (s >= lo) & (s <= hi)
        return m

    def _map_uid_to_index(uid: int):
        fast = uid - 1
        if 0 <= fast < N and int(df.loc[fast, unit_col]) == uid:
            return fast
        hits = np.where(df[unit_col].to_numpy().astype(int) == uid)[0]
        return int(hits[0]) if len(hits) else None

    def _recompute(_=None):
        mask = _current_mask()
        idxs = np.where(mask)[0]
        n = len(idxs)

        # dropdown
        choices = [("— none —", -1)]
        for i in idxs:
            uid = int(df.loc[i, unit_col])
            lab = f"{uid}" + (f" ({df.loc[i, layer_col]})" if layer_col else "")
            choices.append((lab, uid))
        old = unit_dd.value
        unit_dd.options = choices
        unit_dd.value = old if any(v == old for _, v in choices) else -1

        # highlight
        hl_idx = None if unit_dd.value == -1 else _map_uid_to_index(int(unit_dd.value))

        status.value = f"<b>Selected:</b> {n} unit(s)" + (f" • highlighted: {unit_dd.value}" if unit_dd.value != -1 else "")

        tuning_img.value = _plot_tuning(angles, tc_curves, idxs, hl_idx)
        psth_img.value   = _plot_psth(t, psth_curves, idxs, hl_idx)

        if unit_dd.value != -1 and hl_idx is not None:
            row = df.iloc[hl_idx]
            rows = "".join(
                f"<tr><th style='text-align:left;padding-right:8px'>{k}</th><td>{v}</td></tr>"
                for k, v in row.items()
            )
            features_html.value = f"<table>{rows}</table>"
        else:
            features_html.value = "<i>No unit selected.</i>"

    def _reset(_=None):
        if layer_ms:
            layer_ms.value = tuple(layer_ms.options)
        for sl in sliders.values():
            sl.value = (sl.min, sl.max)

    # wire
    for sl in sliders.values():
        sl.observe(_recompute, names="value")
    if layer_ms:
        layer_ms.observe(_recompute, names="value")
    unit_dd.observe(_recompute, names="value")
    reset_btn.on_click(_reset)

    _recompute()
    return app

def main():
    return build_app()
