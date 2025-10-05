from __future__ import annotations
from utils_lazy import get_np, get_pd

# Expected data file locations (first found wins)
DATA_PATHS = (
    ("content/data/meta.csv", "content/data/tuning_curves.npz", "content/data/psth.npz"),
    ("data/meta.csv",         "data/tuning_curves.npz",         "data/psth.npz"),
    ("/files/data/meta.csv",  "/files/data/tuning_curves.npz",  "/files/data/psth.npz"),
    ("/drive/data/meta.csv",  "/drive/data/tuning_curves.npz",  "/drive/data/psth.npz"),
)

def load_all():
    """Return:
        (df, meta_p), (angles, tc_curves, tc_p), (t, psth_curves, psth_p)
    """
    np, pd = get_np(), get_pd()
    last_err = None
    for meta_p, tc_p, psth_p in DATA_PATHS:
        try:
            df    = pd.read_csv(meta_p)
            tc    = np.load(tc_p, allow_pickle=True)
            psth  = np.load(psth_p, allow_pickle=True)
            angles       = np.asarray(tc["angles"])
            tc_curves    = np.asarray(tc["curves"])
            t            = np.asarray(psth["t"])
            psth_curves  = np.asarray(psth["curves"])
            return (df, meta_p), (angles, tc_curves, tc_p), (t, psth_curves, psth_p)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load data from known paths. Last error: {last_err!r}")

def unit_col(df) -> str:
    for c in ("unit_id","unit","id","neuron_id","Unit","Unit_id"):
        if c in df.columns:
            return c
    return ""

def numeric_feature_columns(df, exclude_cols=()):
    """Coerce numeric-like columns and return their names."""
    pd = get_pd()
    numeric_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.95:
            df[c] = s
            numeric_cols.append(c)
    return numeric_cols
