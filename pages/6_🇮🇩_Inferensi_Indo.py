"""
6_🇮🇩_Inferensi_Indo.py
Halaman inferensi untuk subjek Indonesia (pilot study).

Pipeline:
  1. Baca file Tobii XLSX pilot study (sudah di-bundle di folder app).
  2. Pilih subjek dari dropdown.
  3. Build GCN graph dari data gaze mentah.
  4. Fine-tuning ringan (LR=1e-7, 75 epoch) dari tiap pretrained fold
     menggunakan semua subjek lain sebagai data latih (LOO).
  5. Ensemble probabilitas lintas fold → prediksi High Risk / Low Risk.
"""

import os
import random
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch

from utils import inject_custom_css, section_header, divider, render_sidebar
from models import DyslexiaGCN
from indo_utils import parse_tobii_file, build_graph_from_rawdf, build_scanpath_image

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Inferensi Subjek Indonesia",
    page_icon="🇮🇩",
    layout="wide",
)
inject_custom_css()
render_sidebar()

st.markdown('<div class="hero-title">🇮🇩 Inferensi Subjek Indonesia</div>',
            unsafe_allow_html=True)
st.markdown(
    '<p style="color:#8892A0; font-size:1.1rem; max-width:800px;">'
    'Jalankan prediksi risiko disleksia pada subjek <strong>pilot study Indonesia</strong> '
    'menggunakan model DyslexiaGCN yang di-fine-tune dari pretrained weights '
    '(5-fold, Leave-One-Out). Proses ini membutuhkan beberapa menit.'
    '</p>',
    unsafe_allow_html=True,
)
divider()

# ─── Constants ──────────────────────────────────────────────
# __file__ is inside pages/, naik satu level ke root ui_model/
PAGES_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR      = os.path.dirname(PAGES_DIR)
WEIGHTS_DIR   = os.path.join(BASE_DIR, "weights")
EXCEL_PATH    = os.path.join(BASE_DIR, "pilot_study Data export_9.1.2025 (1).xlsx")
SCREEN_W, SCREEN_H = 1680, 1050
MAX_FIXATIONS = 150
K_NEIGHBORS   = 5
GNN_HIDDEN    = 128
EPOCHS        = 75
LR_FINETUNE   = 1e-7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ─── Load & parse Tobii file (cached) ───────────────────────
@st.cache_data(show_spinner="Membaca file pilot study…", ttl=3600)
def load_indo_subjects():
    if not os.path.exists(EXCEL_PATH):
        return None, f"File tidak ditemukan: {EXCEL_PATH}"
    try:
        subjects = parse_tobii_file(EXCEL_PATH, screen_w=SCREEN_W, screen_h=SCREEN_H)
        return subjects, None
    except Exception as e:
        return None, str(e)

all_subjects, load_error = load_indo_subjects()

if load_error:
    st.error(f"❌ Gagal membaca file dataset: `{load_error}`")
    st.stop()

if not all_subjects:
    st.warning("⚠️ Tidak ada partisipan yang ditemukan di file Excel.")
    st.stop()

subject_names = sorted(all_subjects.keys())

# ─── Get available GCN fold weights ─────────────────────────
available_folds = [
    os.path.join(WEIGHTS_DIR, f"gcn_weights_fold_{i}.pth")
    for i in range(5)
    if os.path.exists(os.path.join(WEIGHTS_DIR, f"gcn_weights_fold_{i}.pth"))
]

# ─── UI: Subject Selection ───────────────────────────────────
section_header("Pilih Subjek & Konfigurasi", "⚙️")

col_sub, col_info = st.columns([2, 1])

with col_sub:
    st.markdown(
        f'<div class="info-card" style="margin-bottom:16px;">'
        f'<h4>📂 Dataset Pilot Study</h4>'
        f'<p>Ditemukan <strong>{len(subject_names)}</strong> partisipan · '
        f'<strong>{len(available_folds)}</strong> pretrained GCN fold tersedia</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    selected_subjects = st.multiselect(
        "Pilih subjek yang akan diprediksi",
        options=subject_names,
        default=subject_names[:min(5, len(subject_names))],
        help="Pilih minimal 2 subjek agar fine-tuning Leave-One-Out dapat berjalan.",
        key="indo_subject_select",
    )

with col_info:
    st.markdown(
        '<div class="info-card">'
        '<h4>ℹ️ Alur Inferensi</h4>'
        '<p>'
        '① Parse data gaze mentah<br>'
        '② Build GCN graph tiap subjek<br>'
        '③ Fine-tune dari pretrained (LOO)<br>'
        '④ Ensemble 5 fold<br>'
        '⑤ Prediksi <strong>High Risk / Low Risk</strong>'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─── Validation ─────────────────────────────────────────────
if len(selected_subjects) < 2:
    st.warning("⚠️ Pilih minimal **2 subjek** agar proses Leave-One-Out dapat berjalan.")
    st.stop()

if not available_folds:
    st.error("❌ Tidak ada file weights GCN (`gcn_weights_fold_*.pth`) di folder `weights/`.")
    st.stop()

divider()

# ─── Run Inference ───────────────────────────────────────────
run_btn = st.button(
    "🚀 Jalankan Inferensi",
    key="run_indo_inference",
    use_container_width=True,
)

if run_btn:
    section_header("Proses Inferensi", "🔄")

    # ── Step 1: Build graphs ──────────────────────────────────
    progress = st.progress(0, text="Membangun graf dari data gaze…")
    set_seed(42)

    graph_cache = {}
    failed_subjects = []

    for idx, subj in enumerate(selected_subjects):
        try:
            raw_df, fix_df = all_subjects[subj]
            g = build_graph_from_rawdf(
                raw_df,
                max_fixations=MAX_FIXATIONS,
                k_neighbors=K_NEIGHBORS,
                screen_w=SCREEN_W,
                screen_h=SCREEN_H,
            )
            # Remove batch attr if present, add placeholder y
            if hasattr(g, "batch"):
                del g.batch
            g.y = torch.tensor([0], dtype=torch.long)  # label unknown (screening)
            graph_cache[subj] = g
        except Exception as e:
            failed_subjects.append((subj, str(e)))
        progress.progress((idx + 1) / len(selected_subjects),
                          text=f"Graph [{idx+1}/{len(selected_subjects)}]: {subj}")

    if failed_subjects:
        for s, err in failed_subjects:
            st.warning(f"⚠️ Gagal membangun graf untuk **{s}**: {err}")

    valid_subjects = list(graph_cache.keys())
    if len(valid_subjects) < 2:
        st.error("❌ Tidak cukup subjek valid untuk menjalankan LOO fine-tuning.")
        st.stop()

    # ── Step 2: LOO fine-tuning per fold ─────────────────────
    per_fold_probs = {subj: [] for subj in valid_subjects}
    n_folds = len(available_folds)
    total_steps = n_folds * len(valid_subjects)
    step_count  = 0

    progress.progress(0, text="Memulai fine-tuning (LOO)…")

    for f_idx, fold_path in enumerate(available_folds):
        fold_name = os.path.basename(fold_path).replace("gcn_weights_", "").replace(".pth", "")
        set_seed(123)

        for test_subj in valid_subjects:
            progress.progress(
                step_count / total_steps,
                text=f"Fold {fold_name} | Fine-tuning LOO untuk: {test_subj}…",
            )

            # ─ Prepare train batch (all others)
            train_graphs = [graph_cache[n].clone() for n in valid_subjects if n != test_subj]
            # Assign dummy labels (0) so CrossEntropyLoss can run
            for g in train_graphs:
                g.y = torch.tensor([0], dtype=torch.long)
            train_batch = Batch.from_data_list(train_graphs).to(device)

            # ─ Load & fine-tune model
            model = DyslexiaGCN(in_channels=4, hidden=GNN_HIDDEN, n_classes=2).to(device)
            model.load_state_dict(torch.load(fold_path, map_location=device))

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)

            model.train()
            for _ in range(EPOCHS):
                optimizer.zero_grad()
                out  = model(train_batch.x, train_batch.edge_index, train_batch.batch)
                loss = criterion(out, train_batch.y)
                loss.backward()
                optimizer.step()

            # ─ Inference on test subject
            model.eval()
            with torch.no_grad():
                tg   = graph_cache[test_subj].clone().to(device)
                bidx = torch.zeros(tg.x.size(0), dtype=torch.long, device=device)
                logits = model(tg.x, tg.edge_index, bidx)
                prob   = F.softmax(logits, dim=1)[0, 1].item()

            per_fold_probs[test_subj].append(prob)
            step_count += 1

    progress.progress(1.0, text="✅ Selesai!")

    # ── Step 3: Ensemble & display results ───────────────────
    divider()
    section_header("Hasil Prediksi", "📊")

    results = []
    for subj in valid_subjects:
        probs    = per_fold_probs[subj]
        avg_prob = float(np.mean(probs))
        std_prob = float(np.std(probs))
        pred     = "High Risk 🚨" if avg_prob >= 0.5 else "Low Risk ✅"
        results.append({
            "Subjek":       subj,
            "Probabilitas Rata-rata": avg_prob,
            "Std Dev":      std_prob,
            "Prediksi":     pred,
        })

    df_results = pd.DataFrame(results)

    # ── Summary cards ─────────────────────────────────────────
    n_high = df_results["Prediksi"].str.startswith("High Risk").sum()
    n_low  = len(df_results) - n_high

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">👥 Total Subjek</div>'
            f'<div class="metric-value">{len(df_results)}</div>'
            f'<div class="metric-sub">Berhasil dianalisis</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">🚨 High Risk</div>'
            f'<div class="metric-value" style="color:#FF6B6B;">{n_high}</div>'
            f'<div class="metric-sub">Probabilitas ≥ 50%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">✅ Low Risk</div>'
            f'<div class="metric-value" style="color:#00D4AA;">{n_low}</div>'
            f'<div class="metric-sub">Probabilitas < 50%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-subject cards ─────────────────────────────────────
    for _, row in df_results.iterrows():
        is_high  = row["Prediksi"].startswith("High Risk")
        bar_color = "#FF6B6B" if is_high else "#00D4AA"
        border    = "rgba(255,107,107,0.4)" if is_high else "rgba(0,212,170,0.4)"
        prob_pct  = row["Probabilitas Rata-rata"] * 100

        st.markdown(
            f"""
            <div class="info-card" style="border-color:{border}; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h4 style="margin:0 0 4px 0; color:#E8E8E8;">👤 {row['Subjek']}</h4>
                        <span style="font-size:1.1rem; font-weight:700; color:{bar_color};">
                            {row['Prediksi']}
                        </span>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.8rem; font-weight:800; color:{bar_color};">
                            {prob_pct:.1f}%
                        </div>
                        <div style="font-size:0.75rem; color:#5A6473;">
                            ± {row['Std Dev']*100:.1f}% ({len(available_folds)} fold)
                        </div>
                    </div>
                </div>
                <div style="margin-top:10px; background:rgba(255,255,255,0.05);
                            border-radius:8px; height:8px; overflow:hidden;">
                    <div style="width:{prob_pct:.1f}%; height:100%;
                                background:{bar_color}; border-radius:8px;
                                transition:width 0.5s ease;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Scanpath preview ──────────────────────────────────────
    divider()
    section_header("Pratinjau Scanpath", "👁️")
    st.markdown(
        '<p style="color:#8892A0;">Visualisasi pola gerakan mata subjek yang dipilih. '
        'Garis <span style="color:#FF6B6B;">merah</span> = regresi (gerakan ke kiri).</p>',
        unsafe_allow_html=True,
    )

    preview_sub = st.selectbox(
        "Pilih subjek untuk pratinjau scanpath:",
        options=valid_subjects,
        key="indo_preview_sub",
    )
    if preview_sub:
        _, fix_df_prev = all_subjects[preview_sub]
        fix_clean = fix_df_prev.copy()
        if len(fix_clean) > MAX_FIXATIONS:
            fix_clean = (
                fix_clean.nlargest(MAX_FIXATIONS, "duration_ms")
                .sort_values("seq")
                .reset_index(drop=True)
            )
        if len(fix_clean) >= 2:
            with st.spinner("Merender scanpath…"):
                pil_img = build_scanpath_image(fix_clean, SCREEN_W, SCREEN_H)
            st.image(pil_img, caption=f"Scanpath: {preview_sub}", use_container_width=True)
        else:
            st.warning("Data fiksasi tidak cukup untuk merender scanpath.")

    # ── Raw data table ────────────────────────────────────────
    divider()
    section_header("Tabel Hasil Lengkap", "🗃️")
    display_df = df_results.copy()
    display_df["Probabilitas Rata-rata"] = display_df["Probabilitas Rata-rata"].map(
        lambda v: f"{v*100:.2f}%"
    )
    display_df["Std Dev"] = display_df["Std Dev"].map(lambda v: f"{v*100:.2f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Per-fold probability detail ────────────────────────────
    with st.expander("📈 Detail Probabilitas per Fold"):
        fold_names = [
            os.path.basename(p).replace("gcn_weights_", "").replace(".pth", "")
            for p in available_folds
        ]
        fold_df = pd.DataFrame(
            {subj: per_fold_probs[subj] for subj in valid_subjects},
            index=fold_names,
        ).T
        fold_df.index.name = "Subjek"
        fold_df = fold_df.applymap(lambda v: f"{v*100:.1f}%")
        st.dataframe(fold_df, use_container_width=True)

    st.caption(
        f"⚡ Inferensi selesai · Device: {device} · "
        f"{len(available_folds)} fold pretrained GCN · "
        f"Fine-tune: {EPOCHS} epoch, LR={LR_FINETUNE}"
    )

divider()
st.caption("Catatan: Hasil ini bersifat eksperimental (pilot study). "
           "Diagnosis klinis hanya dapat dilakukan oleh profesional berwenang.")
