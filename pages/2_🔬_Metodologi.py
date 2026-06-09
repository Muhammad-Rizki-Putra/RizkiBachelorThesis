"""
Metodologi Page
"""
import streamlit as st
from utils import inject_custom_css, section_header, divider, render_sidebar

st.set_page_config(page_title="Metodologi", page_icon="🔬", layout="wide")
inject_custom_css()
render_sidebar()

st.markdown('<div class="hero-title">🔬 Metodologi Penelitian</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#8892A0; font-size:1.1rem; max-width:800px;">'
    'Penelitian ini menggunakan tiga cabang pemodelan yang berbeda: Tabular, Citra (CNN), dan Graf (GCN) '
    'untuk mendeteksi disleksia berdasarkan data pergerakan mata.'
    '</p>', 
    unsafe_allow_html=True
)
divider()

# ─── Data Pipeline Awal ──────────────────────────────────────────
section_header("1. Pemrosesan Data Awal", "🛠️")

st.markdown(
    """
    <div class="pipeline-step">
        <div class="step-num">Langkah 1</div>
        <h4>Data Gaze Mentah & Smoothing</h4>
        <p>
            Data input awal berupa <strong>Data Gaze Mentah</strong> (Time-series koordinat x,y dan diameter pupil). 
            Data ini kemudian melalui tahap <strong>Smoothing Kecepatan</strong> menggunakan 
            Filter Savitzky-Golay (Window 7, Orde 2) untuk mengurangi noise.
        </p>
    </div>
    
    <div class="pipeline-step">
        <div class="step-num">Langkah 2</div>
        <h4>Ekstraksi Event (Algoritma I-DT & I-VT)</h4>
        <p>
            Data yang telah diperhalus diproses oleh dua algoritma deteksi event:<br>
            • <strong>Algoritma I-DT</strong> (Threshold Dispersi) untuk <em>Deteksi Fiksasi</em>.<br>
            • <strong>Algoritma I-VT</strong> (Threshold Kecepatan) untuk <em>Deteksi Sakade & Regresi</em>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

divider()

# ─── Tiga Cabang Pre-processing ────────────────────────────────────
section_header("2. Persiapan Data Tiga Cabang", "🔀")

col_tab, col_cit, col_graf = st.columns(3)

with col_tab:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Cabang Tabular\n*(Untuk Machine Learning Klasik)*")
    
    import os
    img_tab = os.path.join(os.path.dirname(__file__), "..", "pic", "preproses", "tabular.png")
    if os.path.exists(img_tab):
        st.image(img_tab, use_container_width=True)
        
    st.markdown(
        """
        **1. Ekstraksi Fitur:**
        Menghasilkan 41 fitur statistik okulomotor (per-task & cross-task).
        
        **2. Normalisasi:**
        Dilakukan menggunakan `StandardScaler`.
        
        **3. Seleksi Fitur:**
        Menggunakan metode RFECV dengan Estimator `Logistic Regression`.
        
        **4. Siap Training:**
        Terpilih 4 Fitur Optimal Berbasis Regresi yang siap dilatih.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_cit:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Cabang Citra\n*(Untuk CNN / ResNet / EfficientNet)*")
    
    img_cit = os.path.join(os.path.dirname(__file__), "..", "pic", "preproses", "citra.png")
    if os.path.exists(img_cit):
        st.image(img_cit, use_container_width=True)
        
    st.markdown(
        """
        **1. Pre-processing Visual:**
        Menggunakan rata-rata koordinat mata & Rolling median window 5.
        
        **2. Rendering Awal:**
        Menggambar jejak pada layar berukuran 1680x1050 px dengan latar hitam.
        
        **3. Protokol Red/White:**
        Visualisasi Segmen Regresi sebagai Garis Merah, dan gerakan Progresif transparan.
        
        **4. Siap Training:**
        Dihasilkan Citra Scanpath berdimensi 224x224 px.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_graf:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🕸️ Cabang Graf\n*(Untuk Graph Convolutional Network)*")
    
    img_graf = os.path.join(os.path.dirname(__file__), "..", "pic", "preproses", "graf.png")
    if os.path.exists(img_graf):
        st.image(img_graf, use_container_width=True)
        
    st.markdown(
        """
        **1. Atribut Simpul (Node):**
        1 Fiksasi direpresentasikan sebagai 1 Node. Fiturnya: cx, cy, durasi, pupil.
        
        **2. Perhitungan Jarak:**
        Menghitung kedekatan spasial antar koordinat fiksasi.
        
        **3. Konstruksi Tepi (Edge):**
        Menghubungkan node menggunakan Algoritma k-Nearest Neighbor (k=5).
        
        **4. Siap Training:**
        Dihasilkan Graf Fiksasi Relasional (Edge Index & Atribut).
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

divider()

# ─── Arsitektur Model ──────────────────────────────────────────────
section_header("3. Arsitektur Model", "🏗️")

col_model_tab, col_model_cit, col_model_graf = st.columns(3)

with col_model_tab:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 1. Tabular (Baseline)")
    st.markdown(
        """
        Pendekatan paling dasar yang menggunakan algoritma machine learning tradisional.
        
        - **Input:** 4 Fitur Regresi (1D).
        - **Algoritma Klasik:** XGBoost / Random Forest / SVM / Logistic Regression.
        - **Output:** Logits Klasifikasi Biner (High-Risk / Low-Risk).
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_model_cit:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 2. Deep Learning Citra")
    st.markdown(
        """
        Menggunakan Tensor Input (Citra Scanpath Red/White) Dimensi 3x224x224. Ada dua jalur:
        
        **A. Simple 3-Layer CNN (Train from scratch):**
        - 3x Conv Blocks (Conv2d, BatchNorm2d, MaxPool2d)
        - AdaptiveAvgPool2d(1) -> Classifier Head
        
        **B. Transfer Learning:**
        - Feature Extractor: Pre-trained ImageNet (ResNet18 atau EffNet-B0).
        - Custom Classifier Head (Dropout & Linear).
        - Regularization Constraint: CrossEntropyLoss dengan label_smoothing=0.1.
        
        **Output:** Logits Klasifikasi Biner.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_model_graf:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 3. DyslexiaGCN")
    st.markdown(
        """
        Pemrosesan graf memanfaatkan PyTorch Geometric.
        
        - **Input Graf:** Node Features (cx, cy, log_duration, seq_pos) & Edge Index (k-NN, k=5).
        - **3x Graph Convolution Blocks:** GCNConv (128 hidden channels) -> BatchNorm1d -> ReLU.
        - **Pooling:** Global Mean Pooling (Agregasi tingkat subjek).
        - **MLP Classifier Head:** Rangkaian Linear layers dengan ReLU & Dropout.
        - **Output:** Logits Klasifikasi Biner (High-Risk / Low-Risk).
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
