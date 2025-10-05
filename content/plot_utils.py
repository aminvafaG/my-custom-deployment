from __future__ import annotations
import io
from utils_lazy import get_np, get_plt

def _to_png(fig, dpi=144):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    get_plt().close(fig)
    return buf.getvalue()

def plot_tuning(angles, tc_curves, idxs, highlight_idx=None):
    np, plt = get_np(), get_plt()
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

def plot_psth(t, psth_curves, idxs, highlight_idx=None):
    np, plt = get_np(), get_plt()
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
