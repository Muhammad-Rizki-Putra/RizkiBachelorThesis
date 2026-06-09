"""
Eksplorasi Data Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    inject_custom_css,
    section_header,
    divider,
    get_plotly_layout,
    load_labels,
    get_subject_list,
    load_subject_data,
    load_aggregated_fixation_stats,
    load_aggregated_saccade_stats,
    load_aggregated_metrics_stats,
    PLOTLY_COLORS,
    TASK_MAP,
    TASK_DISPLAY,
    render_sidebar,
)

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(page_title="Eksplorasi Data", page_icon="📊", layout="wide")
inject_custom_css()
render_sidebar()

st.markdown('<div class="hero-title">📊 Eksplorasi Data</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#8892A0; font-size:1.1rem; max-width:800px;">'
    'Jelajahi dataset eye-tracking antar semua subjek dan tugas membaca. '
    'Analisis pola fiksasi, dinamika saccade, dan lihat distribusi data mentah.'
    '</p>', 
    unsafe_allow_html=True
)
divider()

# ─── Data Loading ───────────────────────────────────────────
with st.spinner("Memuat statistik dataset agregat..."):
    labels_df = load_labels()
    fix_stats = load_aggregated_fixation_stats()
    sacc_stats = load_aggregated_saccade_stats()
    metrics_stats = load_aggregated_metrics_stats()

# ─── Overview Section ───────────────────────────────────────
section_header("Ringkasan Dataset", "📈")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("#### Distribusi Kelas")
    fig_class = px.pie(
        labels_df, 
        names='label', 
        hole=0.6,
        color='label',
        color_discrete_map={'non-dyslexic': PLOTLY_COLORS[0], 'dyslexic': PLOTLY_COLORS[1]}
    )
    fig_class.update_layout(**get_plotly_layout(height=300))
    fig_class.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_class, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("#### Jumlah Fiksasi Berdasarkan Tugas & Kelas")
    if not fix_stats.empty:
        fig_bar = px.box(
            fix_stats, 
            x="task_short", 
            y="n_fixations", 
            color="label",
            color_discrete_map={'non-dyslexic': PLOTLY_COLORS[0], 'dyslexic': PLOTLY_COLORS[1]},
            labels={"task_short": "Tugas", "n_fixations": "Jumlah Fiksasi", "label": "Kelas"}
        )
        fig_bar.update_layout(**get_plotly_layout(height=300, boxmode='group'))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Statistik fiksasi belum termuat sepenuhnya.")
    st.markdown('</div>', unsafe_allow_html=True)

divider()

# ─── Subject-Level Exploration ──────────────────────────────
section_header("Analisis Tingkat Subjek", "👤")

st.markdown(
    """
    Pilih subjek dan tugas untuk memvisualisasikan pola eye-tracking spesifik mereka.
    """
)

col_cat, col_sub, col_task = st.columns(3)

with col_cat:
    kategori_opsi = ["Semua Kategori", "Disleksia", "Non-Disleksia"]
    selected_kategori = st.selectbox("Pilih Kategori", kategori_opsi)

with col_sub:
    # Filter subjects based on category selection
    if selected_kategori == "Disleksia":
        filtered_subjects = labels_df[labels_df['label'] == 'dyslexic']['subject_id'].tolist()
    elif selected_kategori == "Non-Disleksia":
        filtered_subjects = labels_df[labels_df['label'] == 'non-dyslexic']['subject_id'].tolist()
    else:
        filtered_subjects = labels_df['subject_id'].tolist()
        
    selected_subject = st.selectbox("Pilih ID Subjek", sorted(filtered_subjects))
    
    # Display badge
    if selected_subject:
        subject_label = labels_df[labels_df['subject_id'] == selected_subject]['label'].values[0]
        badge_class = "badge-dyslexic" if subject_label == "dyslexic" else "badge-non-dyslexic"
        st.markdown(f'Kelas: <span class="badge {badge_class}">{subject_label.upper()}</span>', unsafe_allow_html=True)

with col_task:
    selected_task_display = st.selectbox("Pilih Tugas Membaca", list(TASK_DISPLAY.values()))
    # Reverse lookup key
    selected_task_key = [k for k, v in TASK_DISPLAY.items() if v == selected_task_display][0]


# Load specific subject data
with st.spinner(f"Memuat data untuk Subjek {selected_subject} - {selected_task_key}..."):
    fix_data = load_subject_data(selected_subject, selected_task_key, "fixations")
    raw_data = load_subject_data(selected_subject, selected_task_key, "raw")

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["👁️ Jalur Tatapan (Scanpath)", "⏱️ Distribusi Fiksasi", "🗃️ Penampil Data Mentah"])

with tab1:
    if fix_data is not None and not fix_data.empty:
        st.markdown("#### Jalur Fiksasi (Gaze Trajectory)")
        st.markdown("Memvisualisasikan urutan dan durasi fiksasi.")
        
        # Plotly scatter for fixations
        fig_scanpath = go.Figure()
        
        # Draw lines between fixations
        fig_scanpath.add_trace(go.Scatter(
            x=fix_data['fix_x'],
            y=fix_data['fix_y'],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.2)', width=1),
            showlegend=False
        ))
        
        # Draw fixations as circles, size based on duration
        color_val = PLOTLY_COLORS[1] if subject_label == "dyslexic" else PLOTLY_COLORS[0]
        fig_scanpath.add_trace(go.Scatter(
            x=fix_data['fix_x'],
            y=fix_data['fix_y'],
            mode='markers+text',
            marker=dict(
                size=fix_data['duration_ms'] / 20,  # scale size
                color=color_val,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=fix_data.index,
            textposition="top center",
            name="Fiksasi"
        ))
        
        layout_dict = get_plotly_layout(height=500)
        layout_dict['xaxis'].update(autorange="reversed")
        layout_dict['yaxis'].update(autorange="reversed")
        fig_scanpath.update_layout(**layout_dict)
        st.plotly_chart(fig_scanpath, use_container_width=True)
    else:
        st.warning("Data fiksasi tidak ditemukan untuk subjek/tugas ini.")

with tab2:
    if fix_data is not None and not fix_data.empty:
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            fig_hist = px.histogram(
                fix_data, 
                x="duration_ms", 
                nbins=30,
                color_discrete_sequence=[PLOTLY_COLORS[1] if subject_label == "dyslexic" else PLOTLY_COLORS[0]],
                title="Distribusi Durasi Fiksasi",
                labels={"duration_ms": "Durasi (ms)"}
            )
            fig_hist.update_layout(**get_plotly_layout(height=400))
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_dist2:
            if 'disp_x' in fix_data.columns and 'disp_y' in fix_data.columns:
                fig_disp = px.scatter(
                    fix_data,
                    x="disp_x",
                    y="disp_y",
                    color_discrete_sequence=[PLOTLY_COLORS[1] if subject_label == "dyslexic" else PLOTLY_COLORS[0]],
                    title="Dispersi Fiksasi (X vs Y)"
                )
                fig_disp.update_layout(**get_plotly_layout(height=400))
                st.plotly_chart(fig_disp, use_container_width=True)
    else:
         st.warning("Data fiksasi tidak ditemukan.")

with tab3:
    if fix_data is not None and not fix_data.empty:
        st.markdown("#### Tabel Data Fiksasi")
        st.dataframe(fix_data.head(100), use_container_width=True)
        st.caption(f"Menampilkan 100 baris pertama dari total {len(fix_data)} fiksasi.")
    else:
        st.warning("Data tidak ditemukan.")

