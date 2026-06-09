import numpy as np
import torch
import pandas as pd
from typing import Tuple

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# Constants for extraction (match the training config)
SCREEN_W = 1680
SCREEN_H = 1050
SCREEN_WIDTH_CM = 47.0
VIEWING_DIST_CM = 60.0
PX_PER_DEG = (SCREEN_W / SCREEN_WIDTH_CM) * (VIEWING_DIST_CM * np.tan(np.deg2rad(1)))

FIXATION_MAX_DISP = 1.0
FIXATION_MIN_PTS = 6
MAX_FIXATIONS = 150
K_NEIGHBORS = 5

def extract_fixations(df: pd.DataFrame, fixation_min_pts=FIXATION_MIN_PTS) -> np.ndarray:
    """I-DT fixation extraction for a single subject DataFrame."""
    # Ensure float type and handle European commas if any
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            except:
                pass
                
    if 'avg_x' not in df.columns:
        lx = next((c for c in df.columns if 'left' in c.lower() and 'x' in c.lower()), None)
        rx = next((c for c in df.columns if 'right' in c.lower() and 'x' in c.lower()), None)
        df['avg_x'] = (df[lx].astype(float) + df[rx].astype(float)) / 2 if lx and rx else df.iloc[:, 0]
        
    if 'avg_y' not in df.columns:
        ly = next((c for c in df.columns if 'left' in c.lower() and 'y' in c.lower()), None)
        ry = next((c for c in df.columns if 'right' in c.lower() and 'y' in c.lower()), None)
        df['avg_y'] = (df[ly].astype(float) + df[ry].astype(float)) / 2 if ly and ry else df.iloc[:, 1]
        
    if 'timestamp' not in df.columns:
        ts = next((c for c in df.columns if 'time' in c.lower()), None)
        df['timestamp'] = df[ts].astype(float) if ts else np.arange(len(df)) * 4.0 # 250Hz default
        
    x = df['avg_x'].values
    y = df['avg_y'].values
    ts = df['timestamp'].values
    fixations = []
    i = 0
    while i < len(x) - fixation_min_pts:
        wx = x[i:i + fixation_min_pts]
        wy = y[i:i + fixation_min_pts]
        if np.nanmax(wx) - np.nanmin(wx) <= FIXATION_MAX_DISP * PX_PER_DEG and \
           np.nanmax(wy) - np.nanmin(wy) <= FIXATION_MAX_DISP * PX_PER_DEG:
            cx = np.nanmean(wx)
            cy = np.nanmean(wy)
            dur = ts[min(i + fixation_min_pts - 1, len(ts) - 1)] - ts[i]
            fixations.append([cx, cy, dur, float(len(fixations))])
            i += fixation_min_pts
        else:
            i += 1
    arr = np.array(fixations) if fixations else np.zeros((1, 4))
    return arr

def knn_edges(coords: np.ndarray, k: int) -> torch.Tensor:
    """Build undirected k-NN edge_index from (F, 2) coords."""
    F = len(coords)
    k = min(k, F - 1)
    if k <= 0:
        return torch.zeros((2, 0), dtype=torch.long)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argsort(dist, axis=1)[:, :k]
    src = np.repeat(np.arange(F), k)
    dst = nn_idx.reshape(-1)
    src_all = np.concatenate([src, dst])
    dst_all = np.concatenate([dst, src])
    edge_index = np.unique(np.stack([src_all, dst_all], axis=0), axis=1)
    return torch.tensor(edge_index, dtype=torch.long)

def dataframe_to_graph(df: pd.DataFrame) -> Tuple[Data, np.ndarray]:
    """
    Converts a raw eye-tracking DataFrame into a PyTorch Geometric Data object.
    Also returns the raw coordinates for plotting.
    """
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for graph features.")
        
    nodes = extract_fixations(df)
    
    # Optional capping
    if len(nodes) > MAX_FIXATIONS:
        top_idx = np.argsort(nodes[:, 2])[::-1][:MAX_FIXATIONS]
        top_idx = np.sort(top_idx)
        nodes = nodes[top_idx]
        
    # Copy coords before normalizing to return for plotting
    raw_coords = nodes[:, :2].copy()
    
    # Normalize features exactly as in training
    nodes[:, 0] /= (SCREEN_W + 1e-06)
    nodes[:, 1] /= (SCREEN_H + 1e-06)
    nodes[:, 2] = np.log1p(nodes[:, 2])
    nodes[:, 3] /= (len(nodes) + 1e-06)
    
    coords = nodes[:, :2]
    edge_index = knn_edges(coords, K_NEIGHBORS)
    node_feats = torch.tensor(nodes, dtype=torch.float)
    
    data = Data(x=node_feats, edge_index=edge_index)
    return data, raw_coords