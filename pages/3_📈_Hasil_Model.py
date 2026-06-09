"""
Hasil Model Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    inject_custom_css,
    section_header,
    divider,
    get_plotly_layout,
    metric_card,
    PLOTLY_COLORS,
    render_sidebar,
)

st.set_page_config(page_title="Hasil Model", page_icon="📈", layout="wide")
inject_custom_css()
render_sidebar()

st.markdown('<div class="hero-title">📈 Hasil & Evaluasi Model</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#8892A0; font-size:1.1rem; max-width:800px;">'
    'Evaluasi komprehensif dari model machine learning yang dilatih untuk memprediksi disleksia. '
    'Bandingkan performa antara pendekatan Tabular dan Citra.'
    '</p>', 
    unsafe_allow_html=True
)
divider()

# ─── Data Hasil Evaluasi Aktual ─────────────────
eval_results = {
    "Tabular (Random Forest)": {"Akurasi": 0.700, "Presisi": 0.706, "Recall": 0.686, "F1-Score": 0.696, "AUC": 0.778},
    "Tabular (SVM)": {"Akurasi": 0.700, "Presisi": 0.719, "Recall": 0.657, "F1-Score": 0.687, "AUC": 0.746},
    "Tabular (Logistic Regression)": {"Akurasi": 0.714, "Presisi": 0.727, "Recall": 0.686, "F1-Score": 0.706, "AUC": 0.759},
    "Tabular (XGBoost)": {"Akurasi": 0.729, "Presisi": 0.735, "Recall": 0.714, "F1-Score": 0.725, "AUC": 0.782},
    "Citra (CNN)": {"Akurasi": 0.8571, "Presisi": 0.7907, "Recall": 0.9714, "F1-Score": 0.8718, "AUC": 0.8571},
    "Citra (ResNet18)": {"Akurasi": 0.8286, "Presisi": 0.8286, "Recall": 0.8286, "F1-Score": 0.8286, "AUC": 0.7910},
    "Citra (EfficientNet)": {"Akurasi": 0.8286, "Presisi": 0.7949, "Recall": 0.8857, "F1-Score": 0.8378, "AUC": 0.8629},
    "Graf (GCN)": {"Akurasi": 0.9000, "Presisi": 0.8889, "Recall": 0.9143, "F1-Score": 0.9014, "AUC": 0.9233},
    "Ensemble (Multimodal)": {"Akurasi": 0.9429, "Presisi": 0.9189, "Recall": 0.9714, "F1-Score": 0.9444, "AUC": 0.9347},
}
df_results = pd.DataFrame(eval_results).T.reset_index().rename(columns={"index": "Model"})

# ─── Top Metrics ────────────────────────────────────────────
st.markdown("### Model dengan Performa Terbaik: Ensemble (Multimodal)")
col1, col2, col3, col4 = st.columns(4)
best = eval_results["Ensemble (Multimodal)"]
with col1:
    metric_card("Akurasi", f"{best['Akurasi']*100:.1f}%", "Ketepatan keseluruhan")
with col2:
    metric_card("Presisi", f"{best['Presisi']*100:.1f}%", "Akurasi prediksi disleksia")
with col3:
    metric_card("Recall", f"{best['Recall']*100:.1f}%", "Tingkat deteksi disleksia sebenarnya")
with col4:
    metric_card("F1-Score", f"{best['F1-Score']*100:.1f}%", "Rata-rata harmonik")

divider()

# ─── Performance Comparison ─────────────────────────────────
section_header("Perbandingan Model", "⚖️")

tab1, tab2, tab3 = st.tabs(["📊 Diagram Batang", "🗃️ Tabel Data", "📈 Akurasi Per Fold"])

with tab1:
    df_melted = df_results.melt(id_vars="Model", var_name="Metrik", value_name="Skor")
    fig_comp = px.bar(
        df_melted, 
        x="Metrik", 
        y="Skor", 
        color="Model", 
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_comp.update_layout(**get_plotly_layout(height=450))
    fig_comp.update_yaxes(range=[0.0, 1.05])
    st.plotly_chart(fig_comp, use_container_width=True)

with tab2:
    st.dataframe(
        df_results.style.format({
            "Akurasi": "{:.2%}",
            "Presisi": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-Score": "{:.2%}",
            "AUC": "{:.2%}"
        }),
        use_container_width=True
    )

with tab3:
    # Data akurasi per fold untuk model Deep Learning & Graf (Non-Tabular)
    fold_data = {
        "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        "Citra (CNN)": [0.7143, 0.9286, 0.7857, 1.0000, 0.8571],
        "Citra (ResNet18)": [0.6429, 1.0000, 0.8571, 0.7857, 0.8571],
        "Citra (EfficientNet)": [0.6429, 1.0000, 0.8571, 0.8571, 0.7857],
        "Graf (GCN)": [0.9286, 1.0000, 0.7857, 0.8571, 0.9286],
        "Ensemble (Multimodal)": [0.9286, 1.0000, 0.8571, 1.0000, 0.9286]
    }
    df_fold = pd.DataFrame(fold_data)
    df_fold_melted = df_fold.melt(id_vars="Fold", var_name="Model", value_name="Akurasi")
    
    fig_fold = px.line(
        df_fold_melted, 
        x="Fold", 
        y="Akurasi", 
        color="Model", 
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_fold.update_layout(
        **get_plotly_layout(height=450),
        yaxis_title="Akurasi",
        xaxis_title="Cross-Validation Fold"
    )
    fig_fold.update_yaxes(range=[0.5, 1.05], tickformat=".0%")
    st.plotly_chart(fig_fold, use_container_width=True)

divider()

# ─── Confusion Matrix & ROC (Mocked Visuals) ────────────────
section_header("Diagnostik Detail", "🔍")

col_cm, col_roc = st.columns(2)

with col_cm:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("#### Matriks Konfusi Aktual")
    
    cm_options = {
        "Ensemble (Multimodal)": "pic/ensemble.png",
        "Graf (GCN)": "pic/gcn.png",
        "Citra (ResNet18)": "pic/resnet.png",
        "Citra (EfficientNet)": "pic/eff.png",
        "Citra (CNN)": "pic/cnn.png"
    }
    
    selected_cm = st.selectbox("Pilih Matriks Konfusi Model:", list(cm_options.keys()))
    
    import os
    img_path = os.path.join(os.path.dirname(__file__), "..", cm_options[selected_cm])
    
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning(f"Gambar tidak ditemukan di: {img_path}")
        
    st.markdown('</div>', unsafe_allow_html=True)

with col_roc:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("#### Kurva ROC")
    
    # Ambil nilai AUC asli dari eval_results untuk model yang dipilih
    auc_val = eval_results[selected_cm]["AUC"]
    
    # Generate kurva ROC dinamis yang secara matematis sesuai dengan nilai AUC
    # Menggunakan fungsi tpr = 1 - (1 - fpr)^k dimana k = AUC / (1 - AUC)
    fpr = np.linspace(0, 1, 100)
    k = auc_val / (1.0 - auc_val) if auc_val < 1.0 else 999
    tpr = 1.0 - (1.0 - fpr) ** k
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, 
        name=f"{selected_cm} (AUC = {auc_val:.2f})", 
        line=dict(color='#00D4AA', width=3)
    ))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Tebakan Acak', line=dict(color='gray', dash='dash')))
    
    fig_roc.update_layout(
        **get_plotly_layout(height=350),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Catatan: Metrik tabel didasarkan pada hasil evaluasi akhir. Matriks konfusi menampilkan gambar riwayat model dan Kurva ROC merupakan simulasi visual yang dinamis mengikuti nilai AUC aktual masing-masing model.")
