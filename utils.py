"""
Utility functions for the Dyslexia Detection Streamlit App.
Handles data loading, CSS injection, and common chart configurations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "model" / "dataset"
DATA_DIR = DATASET_DIR / "data"
LABEL_FILE = DATASET_DIR / "dyslexia_class_label.csv"

TASK_MAP = {
    "T1": "Syllables",
    "T4": "Meaningful_Text",
    "T5": "Pseudo_Text",
}

TASK_DISPLAY = {
    "T1_Syllables": "T1 — Suku Kata",
    "T4_Meaningful_Text": "T4 — Teks Bermakna",
    "T5_Pseudo_Text": "T5 — Teks Semu",
}

# ─────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#00D4AA",
    "secondary": "#4A90D9",
    "accent": "#FF6B6B",
    "warning": "#FFD93D",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1F2E",
    "bg_card_hover": "#242B3D",
    "text": "#E8E8E8",
    "text_muted": "#8892A0",
    "border": "rgba(255,255,255,0.08)",
    "dyslexic": "#FF6B6B",
    "non_dyslexic": "#00D4AA",
}

PLOTLY_COLORS = [COLORS["non_dyslexic"], COLORS["dyslexic"]]
PLOTLY_TASK_COLORS = ["#00D4AA", "#4A90D9", "#FF6B6B"]


# ─────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────
def inject_custom_css():
    """Inject premium dark-theme CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* ── Import Font ────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Global ─────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        .stApp {
            background: linear-gradient(135deg, #0E1117 0%, #151B28 50%, #0E1117 100%);
        }

        /* ── Sidebar ────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #00D4AA !important;
        }

        /* ── Metric Cards ───────────────────────────── */
        .metric-card {
            background: linear-gradient(135deg, rgba(26,31,46,0.9) 0%, rgba(36,43,61,0.7) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 24px;
            margin: 8px 0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        }

        .metric-card:hover {
            border-color: rgba(0,212,170,0.3);
            box-shadow: 0 8px 32px rgba(0,212,170,0.1);
            transform: translateY(-2px);
        }

        .metric-value {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00D4AA, #4A90D9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 4px 0;
            line-height: 1.1;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #8892A0;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .metric-sub {
            font-size: 0.8rem;
            color: #5A6473;
            margin-top: 4px;
        }

        /* ── Section Headers ────────────────────────── */
        .section-header {
            font-size: 1.6rem;
            font-weight: 700;
            color: #E8E8E8;
            margin: 32px 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid rgba(0,212,170,0.3);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* ── Hero Banner ────────────────────────────── */
        .hero-container {
            background: linear-gradient(135deg, rgba(0,212,170,0.08) 0%, rgba(74,144,217,0.08) 50%, rgba(255,107,107,0.05) 100%);
            border: 1px solid rgba(0,212,170,0.15);
            border-radius: 20px;
            padding: 48px 40px;
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
        }

        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%);
            border-radius: 50%;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00D4AA 0%, #4A90D9 50%, #00D4AA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            color: #8892A0;
            line-height: 1.6;
            max-width: 700px;
        }

        /* ── Info Card ──────────────────────────────── */
        .info-card {
            background: rgba(26,31,46,0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 20px 24px;
            margin: 12px 0;
        }

        .info-card h4 {
            color: #00D4AA;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .info-card p {
            color: #B0B8C4;
            line-height: 1.6;
            margin: 0;
        }

        /* ── Badge ──────────────────────────────────── */
        .badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        .badge-dyslexic {
            background: rgba(255,107,107,0.15);
            color: #FF6B6B;
            border: 1px solid rgba(255,107,107,0.3);
        }

        .badge-non-dyslexic {
            background: rgba(0,212,170,0.15);
            color: #00D4AA;
            border: 1px solid rgba(0,212,170,0.3);
        }

        /* ── Pipeline Step ──────────────────────────── */
        .pipeline-step {
            background: rgba(26,31,46,0.7);
            border: 1px solid rgba(255,255,255,0.06);
            border-left: 4px solid #00D4AA;
            border-radius: 0 12px 12px 0;
            padding: 20px 24px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }

        .pipeline-step:hover {
            border-left-color: #4A90D9;
            background: rgba(36,43,61,0.7);
        }

        .pipeline-step .step-num {
            font-size: 0.75rem;
            color: #00D4AA;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .pipeline-step h4 {
            color: #E8E8E8;
            margin: 4px 0 8px 0;
            font-weight: 600;
        }

        .pipeline-step p {
            color: #8892A0;
            margin: 0;
            line-height: 1.5;
            font-size: 0.9rem;
        }

        /* ── Tabs ───────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(26,31,46,0.5);
            border-radius: 12px;
            padding: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            color: #8892A0;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(0,212,170,0.15) !important;
            color: #00D4AA !important;
        }

        /* ── Dataframe ──────────────────────────────── */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* ── Divider ────────────────────────────────── */
        .custom-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0,212,170,0.3), transparent);
            margin: 32px 0;
            border: none;
        }

        /* ── Footer ─────────────────────────────────── */
        .app-footer {
            text-align: center;
            color: #5A6473;
            font-size: 0.8rem;
            padding: 32px 0 16px 0;
            border-top: 1px solid rgba(255,255,255,0.04);
            margin-top: 48px;
        }

        /* ── Hide Streamlit Branding ────────────────── */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: str = "", icon: str = ""):
    """Render a glassmorphism metric card."""
    icon_html = f"<span style='font-size:1.5rem;margin-right:6px;'>{icon}</span>" if icon else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon_html}{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(text: str, icon: str = ""):
    """Render a styled section header."""
    st.markdown(
        f'<div class="section-header">{icon} {text}</div>',
        unsafe_allow_html=True,
    )


def divider():
    """Render a gradient divider."""
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


def footer():
    """Render the app footer."""
    st.markdown(
        """
        <div class="app-footer">
            Deteksi Disleksia via Eye-Tracking · Penelitian Skripsi oleh Muhammad Rizki Putra · 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the standard sidebar with logo and author info across all pages."""
    import os
    with st.sidebar:
        logo_path = BASE_DIR / "pic" / "logo-unpad1.webp"
        st.markdown('<div style="padding: 8px 0;"></div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1, 1.2, 1.2, 1])
        with col2:
            if logo_path.exists():
                st.image(str(logo_path), use_container_width=True)
        with col3:
            st.markdown('<div style="text-align:center; font-size: 2.5rem; line-height: 1.1; margin-top: -2px;">🧠</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div style="text-align:center; padding: 0 0 8px 0;">
                <h2 style="margin:8px 0 2px 0; font-weight:800; 
                    background: linear-gradient(135deg, #00D4AA, #4A90D9);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;">
                    DysDetect
                </h2>
                <p style="color:#5A6473; font-size:0.8rem; margin:0;">
                    Aplikasi Riset Eye-Tracking
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        divider()

        st.markdown(
            """
            <div style="padding: 8px 0;">
                <p style="color:#8892A0; font-size:0.78rem; line-height:1.6;">
                    <strong style="color:#B0B8C4;">Penelitian oleh</strong><br>
                    Muhammad Rizki Putra<br><br>
                    <strong style="color:#B0B8C4;">Tahun</strong><br>
                    2026
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# PLOTLY LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────
def get_plotly_layout(title: str = "", height: int = 450, **kwargs):
    """Return a consistent dark-theme Plotly layout dict."""
    layout = dict(
        title=dict(
            text=title,
            font=dict(size=16, color="#E8E8E8", family="Inter"),
            x=0.02,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,46,0.4)",
        font=dict(family="Inter", color="#8892A0", size=12),
        height=height,
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.06)",
            borderwidth=1,
            font=dict(color="#B0B8C4"),
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.06)",
        ),
    )
    layout.update(kwargs)
    return layout


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_labels() -> pd.DataFrame:
    """Load dyslexia class labels."""
    return pd.read_csv(LABEL_FILE)


@st.cache_data(ttl=3600)
def get_subject_list():
    """Get a list of available subject IDs from the labels file."""
    labels = load_labels()
    return labels["subject_id"].tolist()


@st.cache_data(ttl=3600)
def load_subject_data(subject_id: int, task_key: str, data_type: str) -> pd.DataFrame | None:
    """
    Load a specific data file for a subject.
    
    Args:
        subject_id: Subject ID (e.g. 1003)
        task_key: Task key (e.g. 'T1_Syllables')
        data_type: One of 'fixations', 'saccades', 'metrics', 'raw'
    
    Returns:
        DataFrame or None if file not found.
    """
    filename = f"Subject_{subject_id}_{task_key}_{data_type}.csv"
    filepath = DATA_DIR / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    return None


@st.cache_data(ttl=3600)
def load_aggregated_fixation_stats() -> pd.DataFrame:
    """
    Load and aggregate fixation statistics across all subjects & tasks.
    Returns a DataFrame with per-subject, per-task summary stats.
    """
    labels = load_labels()
    records = []

    for _, row in labels.iterrows():
        sid = row["subject_id"]
        label = row["label"]

        for task_key_short, task_name in TASK_MAP.items():
            task_key = f"{task_key_short}_{task_name}"
            fix_df = load_subject_data(sid, task_key, "fixations")
            if fix_df is not None and not fix_df.empty:
                records.append({
                    "subject_id": sid,
                    "label": label,
                    "task": task_key,
                    "task_short": task_key_short,
                    "n_fixations": len(fix_df),
                    "mean_fix_duration": fix_df["duration_ms"].mean(),
                    "median_fix_duration": fix_df["duration_ms"].median(),
                    "std_fix_duration": fix_df["duration_ms"].std(),
                    "total_fix_duration": fix_df["duration_ms"].sum(),
                    "mean_disp_x": fix_df["disp_x"].mean() if "disp_x" in fix_df.columns else np.nan,
                    "mean_disp_y": fix_df["disp_y"].mean() if "disp_y" in fix_df.columns else np.nan,
                })

    return pd.DataFrame(records)


@st.cache_data(ttl=3600)
def load_aggregated_saccade_stats() -> pd.DataFrame:
    """
    Load and aggregate saccade statistics across all subjects & tasks.
    """
    labels = load_labels()
    records = []

    for _, row in labels.iterrows():
        sid = row["subject_id"]
        label = row["label"]

        for task_key_short, task_name in TASK_MAP.items():
            task_key = f"{task_key_short}_{task_name}"
            sacc_df = load_subject_data(sid, task_key, "saccades")
            if sacc_df is not None and not sacc_df.empty:
                records.append({
                    "subject_id": sid,
                    "label": label,
                    "task": task_key,
                    "task_short": task_key_short,
                    "n_saccades": len(sacc_df),
                    "mean_sacc_duration": sacc_df["duration_ms"].mean(),
                    "mean_sacc_amplitude": sacc_df["ampl"].mean() if "ampl" in sacc_df.columns else np.nan,
                    "mean_avg_velocity": sacc_df["avg_vel"].mean() if "avg_vel" in sacc_df.columns else np.nan,
                    "mean_peak_velocity": sacc_df["peak_vel"].mean() if "peak_vel" in sacc_df.columns else np.nan,
                })

    return pd.DataFrame(records)


@st.cache_data(ttl=3600)
def load_aggregated_metrics_stats() -> pd.DataFrame:
    """
    Load trial-level metrics across all subjects & tasks (from the metrics CSV).
    Only extracts the first row per file (trial-level summary).
    """
    labels = load_labels()
    records = []

    for _, row in labels.iterrows():
        sid = row["subject_id"]
        label = row["label"]

        for task_key_short, task_name in TASK_MAP.items():
            task_key = f"{task_key_short}_{task_name}"
            met_df = load_subject_data(sid, task_key, "metrics")
            if met_df is not None and not met_df.empty:
                first = met_df.iloc[0]
                records.append({
                    "subject_id": sid,
                    "label": label,
                    "task": task_key,
                    "task_short": task_key_short,
                    "n_fix_trial": first.get("n_fix_trial", np.nan),
                    "sum_fix_dur_trial": first.get("sum_fix_dur_trial", np.nan),
                    "mean_fix_dur_trial": first.get("mean_fix_dur_trial", np.nan),
                    "dwell_time_trial": first.get("dwell_time_trial", np.nan),
                    "n_sacc_trial": first.get("n_sacc_trial", np.nan),
                    "mean_sacc_ampl_trial": first.get("mean_sacc_ampl_trial", np.nan),
                    "ratio_progress_regress": first.get("ratio_progress_regress_trial", np.nan),
                    "n_regress_trial": first.get("n_regress_trial", np.nan),
                    "n_progress_trial": first.get("n_progress_trial", np.nan),
                })

    return pd.DataFrame(records)
