from __future__ import annotations
import ipywidgets as W

from utils_lazy import get_np
from data_utils import load_all, unit_col, numeric_feature_columns
from plot_utils import plot_tuning, plot_psth

def build_app():
    np = get_np()
    (df, meta_p), (angles, tc_curves, tc_p), (t, psth_curves, psth_p) = load_all()
    df = df.copy()
    N = len(df)

    ucol = unit_col(df)
    if ucol == "":
        df["unit_id"] = np.arange(1, N+1)
        ucol = "unit_id"

    # optional 'layer'
    layer_col = None
    for c in ("layer","Layer","lamina","Lamina"):
        if c in df.columns:
            layer_col = c
            break

    num_cols = numeric_feature_columns(df, exclude_cols=(ucol, layer_col) if layer_col else (ucol,))
    # sliders
    sliders = {}
    for c in num_cols:
        s = df[c].astype(float)
        lo = float(np.nanmin(s))
        hi = float(np.nanmax(s))
        step = (hi - lo)/200.0 if np.isfinite(hi-lo) and (hi-lo) > 0 else 0.01
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
        pane_widths=["340px", "1fr", "0px"],
        pane_heights=["30px", "1fr", "30px"],
    )

    # ---------- logic ----------
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
        if 0 <= fast < N and int(df.loc[fast, ucol]) == uid:
            return fast
        hits = np.where(df[ucol].to_numpy().astype(int) == uid)[0]
        return int(hits[0]) if len(hits) else None

    def _recompute(_=None):
        import numpy as _np  # local alias for args to functions
        idxs = _np.where(_current_mask())[0]
        n = len(idxs)

        # dropdown
        choices = [("— none —", -1)]
        for i in idxs:
            uid = int(df.loc[i, ucol])
            lab = f"{uid}" + (f" ({df.loc[i, layer_col]})" if layer_col else "")
            choices.append((lab, uid))
        old = unit_dd.value
        unit_dd.options = choices
        unit_dd.value = old if any(v == old for _, v in choices) else -1

        # highlight
        hl_idx = None if unit_dd.value == -1 else _map_uid_to_index(int(unit_dd.value))

        status.value = f"<b>Selected:</b> {n} unit(s)" + (f" • highlighted: {unit_dd.value}" if unit_dd.value != -1 else "")

        tuning_img.value = plot_tuning(angles, tc_curves, idxs, hl_idx)
        psth_img.value   = plot_psth(t, psth_curves, idxs, hl_idx)

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
