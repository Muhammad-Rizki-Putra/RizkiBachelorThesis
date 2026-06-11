"""
indo_utils.py
Utility functions khusus untuk inferensi subjek Indonesia (pilot study).
- Parsing file Tobii (.xlsx / .tsv / .csv) multi-partisipan
- Build GCN graph dari raw gaze DataFrame
- Build scanpath image (PIL) dari fixation DataFrame
"""

import io
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ── Screen constants (match training config) ──────────────────
SCREEN_W = 1680
SCREEN_H = 1050
MAX_FIXATIONS = 150
K_NEIGHBORS   = 5


# ─────────────────────────────────────────────────────────────
# 1.  TOBII FILE PARSER
# ─────────────────────────────────────────────────────────────

def _load_tobii_file(filepath: str) -> pd.DataFrame:
    """Read .xlsx/.xlsm or .tsv/.csv into a DataFrame."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        df = pd.read_excel(filepath, sheet_name=0)
    else:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            header_line = f.readline()
        sep = ";" if ";" in header_line else "\t"
        df = pd.read_csv(filepath, sep=sep, low_memory=False,
                         decimal=",", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    return df


def _parse_one_participant(df: pd.DataFrame, subject_id: str,
                           screen_w: int, screen_h: int):
    """Extract raw gaze + fixation DataFrames for one participant."""

    def find_col(*keywords):
        for col in df.columns:
            if all(k.lower() in col.lower() for k in keywords):
                return col
        return None

    def to_float(col_name):
        s = df[col_name].copy()
        if s.dtype == object:
            s = s.str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    # ── Gaze columns ──────────────────────────────────────────
    lx = find_col("gaze point left", "x", "dacs px") or find_col("left", "x")
    rx = find_col("gaze point right", "x", "dacs px") or find_col("right", "x")
    ly = find_col("gaze point left", "y", "dacs px") or find_col("left", "y")
    ry = find_col("gaze point right", "y", "dacs px") or find_col("right", "y")
    ts_col = find_col("recording timestamp") or find_col("timestamp")

    raw_df = pd.DataFrame({
        "gaze_x_left":  to_float(lx) if lx else np.nan,
        "gaze_x_right": to_float(rx) if rx else np.nan,
        "gaze_y_left":  to_float(ly) if ly else np.nan,
        "gaze_y_right": to_float(ry) if ry else np.nan,
    })

    if ts_col:
        ts = to_float(ts_col)
        raw_df["timestamp"] = ts - ts.min()
        if raw_df["timestamp"].max() > 1_000_000:
            raw_df["timestamp"] /= 1000.0
    else:
        raw_df["timestamp"] = np.arange(len(raw_df)) * (1000.0 / 60.0)

    for c in ["gaze_x_left", "gaze_x_right"]:
        raw_df[c] = raw_df[c].clip(0, screen_w)
    for c in ["gaze_y_left", "gaze_y_right"]:
        raw_df[c] = raw_df[c].clip(0, screen_h)

    raw_df["avg_x"] = raw_df[["gaze_x_left", "gaze_x_right"]].mean(axis=1)
    raw_df["avg_y"] = raw_df[["gaze_y_left", "gaze_y_right"]].mean(axis=1)
    raw_df["avg_x"].replace([0.0, -1.0], np.nan, inplace=True)
    raw_df["avg_y"].replace([0.0, -1.0], np.nan, inplace=True)
    raw_df["avg_x"] = raw_df["avg_x"].interpolate(method="linear", limit=5)
    raw_df["avg_y"] = raw_df["avg_y"].interpolate(method="linear", limit=5)
    raw_df = raw_df.dropna(subset=["avg_x", "avg_y"]).reset_index(drop=True)

    # ── Fixation columns ──────────────────────────────────────
    evt_col     = find_col("eye movement type")
    dur_col     = find_col("eye movement event duration")
    fix_x_col   = find_col("fixation point x")
    fix_y_col   = find_col("fixation point y")
    fix_idx_col = find_col("eye movement type index")

    if evt_col and fix_x_col:
        fix_mask = df[evt_col].astype(str).str.strip().str.lower() == "fixation"
        fix_rows = df[fix_mask].copy()
        if fix_idx_col:
            fix_rows = fix_rows.drop_duplicates(subset=[fix_idx_col], keep="first")
        else:
            dup_cols = [fix_x_col, fix_y_col] + ([dur_col] if dur_col else [])
            fix_rows = fix_rows.drop_duplicates(subset=dup_cols, keep="first")

        fix_df = pd.DataFrame({
            "fix_x": pd.to_numeric(
                fix_rows[fix_x_col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            ).values,
            "fix_y": pd.to_numeric(
                fix_rows[fix_y_col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            ).values,
        })
        if ts_col:
            fix_df["start_ms"] = to_float(ts_col).reindex(fix_rows.index).values
        else:
            fix_df["start_ms"] = np.zeros(len(fix_rows))

        if dur_col:
            durs = pd.to_numeric(
                fix_rows[dur_col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            ).values
            fix_df["duration_ms"] = durs
        else:
            fix_df["duration_ms"] = 150.0

        fix_df["end_ms"] = fix_df["start_ms"] + fix_df["duration_ms"]
        fix_df["seq"]    = np.arange(len(fix_df), dtype=float)
        fix_df = fix_df.dropna(subset=["fix_x", "fix_y"]).reset_index(drop=True)
    else:
        # Fallback: rolling median as pseudo-fixations
        x = raw_df["avg_x"].rolling(10, center=True).median()
        y = raw_df["avg_y"].rolling(10, center=True).median()
        ts_arr = raw_df["timestamp"].values
        x_c = x.dropna().values[::10]
        y_c = y.dropna().values[::10]
        n = min(len(x_c), len(y_c))
        fix_df = pd.DataFrame({
            "fix_x":       x_c[:n],
            "fix_y":       y_c[:n],
            "start_ms":    ts_arr[::10][:n],
            "duration_ms": np.full(n, 150.0),
        })
        fix_df["end_ms"] = fix_df["start_ms"] + fix_df["duration_ms"]
        fix_df["seq"]    = np.arange(len(fix_df), dtype=float)

    if len(fix_df) > 0:
        fix_df["fix_x"] = fix_df["fix_x"].clip(0, screen_w)
        fix_df["fix_y"] = fix_df["fix_y"].clip(0, screen_h)

    return raw_df, fix_df


def parse_tobii_file(filepath: str,
                     screen_w: int = SCREEN_W,
                     screen_h: int = SCREEN_H) -> dict:
    """
    Parse a Tobii export file (xlsx / tsv / csv).
    Returns:
        dict  {participant_name: (raw_df, fix_df)}
    """
    df_full = _load_tobii_file(filepath)

    if "Participant name" in df_full.columns:
        participants = df_full["Participant name"].dropna().unique().tolist()
    else:
        participants = [os.path.splitext(os.path.basename(filepath))[0]]
        df_full["Participant name"] = participants[0]

    all_subjects = {}
    for pname in participants:
        sub_df = df_full[df_full["Participant name"] == pname].reset_index(drop=True)
        raw, fix = _parse_one_participant(sub_df, str(pname), screen_w, screen_h)
        all_subjects[str(pname)] = (raw, fix)

    return all_subjects


# ─────────────────────────────────────────────────────────────
# 2.  GRAPH BUILDER  (matches v29 training config)
# ─────────────────────────────────────────────────────────────

def build_graph_from_rawdf(raw_df: pd.DataFrame,
                           max_fixations: int = MAX_FIXATIONS,
                           k_neighbors: int = K_NEIGHBORS,
                           screen_w: int = SCREEN_W,
                           screen_h: int = SCREEN_H) -> Data:
    """
    Build a PyG Data object from a raw gaze DataFrame.
    Feature vector per node: [cx_norm, cy_norm, log1p(dur), seq_norm]
    """
    fixation_min_pts  = 2
    FIXATION_MAX_DISP = 3.5
    PX_PER_DEG        = 37.44

    x_arr = raw_df["avg_x"].values
    y_arr = raw_df["avg_y"].values

    fixations = []
    i = 0
    while i < len(x_arr) - fixation_min_pts:
        wx = x_arr[i: i + fixation_min_pts]
        wy = y_arr[i: i + fixation_min_pts]
        if (np.nanmax(wx) - np.nanmin(wx) <= FIXATION_MAX_DISP * PX_PER_DEG and
                np.nanmax(wy) - np.nanmin(wy) <= FIXATION_MAX_DISP * PX_PER_DEG):
            fixations.append([np.nanmean(wx), np.nanmean(wy),
                               20.0, float(len(fixations))])
            i += fixation_min_pts
        else:
            i += 1

    nodes = np.array(fixations) if fixations else np.zeros((1, 4))
    if len(nodes) > max_fixations:
        nodes = nodes[:max_fixations]

    nodes[:, 0] /= (screen_w + 1e-6)
    nodes[:, 1] /= (screen_h + 1e-6)
    nodes[:, 2]  = np.log1p(nodes[:, 2])
    nodes[:, 3] /= (len(nodes) + 1e-6)

    F = len(nodes)
    k = min(k_neighbors, F - 1)
    if k <= 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        coords = nodes[:, :2]
        diff   = coords[:, None, :] - coords[None, :, :]
        dist   = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        nn_idx = np.argsort(dist, axis=1)[:, :k]
        src    = np.repeat(np.arange(F), k)
        dst    = nn_idx.reshape(-1)
        ei     = np.unique(
            np.stack([np.concatenate([src, dst]),
                      np.concatenate([dst, src])], axis=0),
            axis=1,
        )
        edge_index = torch.tensor(ei, dtype=torch.long)

    node_feats = torch.tensor(nodes, dtype=torch.float32)
    return Data(x=node_feats, edge_index=edge_index)


# ─────────────────────────────────────────────────────────────
# 3.  SCANPATH IMAGE BUILDER
# ─────────────────────────────────────────────────────────────

def build_scanpath_image(fix_df: pd.DataFrame,
                         screen_w: int = SCREEN_W,
                         screen_h: int = SCREEN_H):
    """
    Render a red/white scanpath image from a fixation DataFrame.
    Returns a PIL Image (RGB).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from PIL import Image

    x = fix_df["fix_x"].values.astype(float)
    y = fix_df["fix_y"].values.astype(float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y  = x[valid], y[valid]

    fig, ax = plt.subplots(figsize=(screen_w / 100, screen_h / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(0, screen_w)
    ax.set_ylim(screen_h, 0)
    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    if len(x) >= 2:
        points   = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        dx = np.diff(x)
        colors = [(1., 0., 0., 0.9) if d < -5 else (0., 0., 0., 0.) for d in dx]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=3))

    ax.scatter(x, y, color="white", s=8, alpha=0.15, zorder=3)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0, facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
