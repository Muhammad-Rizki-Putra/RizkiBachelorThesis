"""
🧠 Deteksi Disleksia via Eye-Tracking
Main entry / Home page for the Streamlit application.
"""

import streamlit as st
from utils import inject_custom_css, metric_card, section_header, divider, footer, load_labels, render_sidebar

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Disleksia · Riset Eye-Tracking",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

render_sidebar()


# ─── Hero Section ───────────────────────────────────────────
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">
            Deteksi Disleksia Menggunakan<br>Analisis Data Eye-Tracking
        </div>
        <div class="hero-subtitle">
            Pendekatan machine learning komprehensif untuk mengklasifikasikan pembaca High Risk dan Low Risk 
            melalui pola pergerakan mata yang direkam selama tugas membaca. Penelitian ini memanfaatkan metrik 
            fiksasi, saccade, dan tatapan dari eye-tracking frekuensi tinggi untuk membangun model prediktif 
            skrining dini risiko disleksia.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Key Metrics ────────────────────────────────────────────
labels = load_labels()
n_total = len(labels)
n_dyslexic = len(labels[labels["label"] == "dyslexic"])
n_non_dyslexic = len(labels[labels["label"] == "non-dyslexic"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Total Subjek", str(n_total), "Partisipan eye-tracking", "👥")
with col2:
    metric_card("Low Risk", str(n_non_dyslexic), "Kelompok kontrol (kelas 0)", "✅")
with col3:
    metric_card("High Risk", str(n_dyslexic), "Kelompok risiko tinggi (kelas 1)", "🔍")
with col4:
    metric_card("Tugas Membaca", "3", "T1 · T4 · T5", "📖")

# ─── Research Overview ──────────────────────────────────────
divider()
section_header("Ringkasan Penelitian", "📋")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown(
        """
        <div class="info-card">
            <h4>🎯 Tujuan</h4>
            <p>
                Mengembangkan dan mengevaluasi model machine learning yang mampu mendeteksi risiko disleksia 
                dari data eye-tracking yang direkam selama tiga tugas membaca berbeda. Studi ini 
                menganalisis pola fiksasi, dinamika saccade, dan fitur okulomotor lainnya 
                untuk membedakan pembaca High Risk dan Low Risk.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>📊 Dataset</h4>
            <p>
                Rekaman eye-tracking dari <strong>70 partisipan</strong> (35 High Risk, 35 Low Risk) 
                yang melakukan tiga tugas:<br>
                <strong>T1 — Suku Kata:</strong> Membaca suku kata individu<br>
                <strong>T4 — Teks Bermakna:</strong> Membaca paragraf teks bermakna<br>
                <strong>T5 — Teks Semu:</strong> Membaca paragraf teks kata semu (pseudo-word)<br><br>
                Setiap rekaman menghasilkan data fiksasi, saccade, tatapan mentah, dan metrik yang dihitung.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        """
        <div class="info-card">
            <h4>🔬 Metodologi</h4>
            <p>
                Penelitian ini menerapkan tiga pendekatan pemodelan utama:<br><br>
                <strong>1. Cabang Tabular</strong> — Mengekstrak fitur statistik okulomotor untuk dilatih pada model Machine Learning klasik (XGBoost, dll).<br><br>
                <strong>2. Cabang Citra</strong> — Mengubah jalur tatapan mata menjadi gambar (scanpath) untuk dilatih menggunakan Deep Learning (ResNet18).<br><br>
                <strong>3. Cabang Graf</strong> — Memetakan fiksasi sebagai simpul jaringan untuk dianalisis melalui Graph Convolutional Network (DyslexiaGCN).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>🧪 Fitur Utama yang Dianalisis</h4>
            <p>
                • Jumlah & durasi fiksasi<br>
                • Amplitudo & kecepatan saccade<br>
                • Waktu singgah (dwell time) per area minat<br>
                • Jumlah regresi (di dalam/antar baris)<br>
                • Rasio progres-regresi<br>
                • Dispersi tatapan (x, y)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Navigation Guide ───────────────────────────────────────
divider()
section_header("Jelajahi Aplikasi", "🧭")

nav_cols = st.columns(5)

nav_items = [
    ("📊", "Eksplorasi Data", "Jelajahi dan visualisasikan data eye-tracking antar subjek dan tugas."),
    ("🔬", "Metodologi", "Pahami alur penelitian dari pengumpulan data hingga pelatihan model."),
    ("📈", "Hasil Model", "Lihat metrik performa klasifikasi, matriks konfusi, dan perbandingan."),
    ("🔍", "Prediksi", "Uji model dengan data sampel atau unggah CSV Anda sendiri untuk prediksi."),
    ("ℹ️", "Tentang", "Detail penelitian, informasi penulis, dan referensi."),
]

for col, (icon, title, desc) in zip(nav_cols, nav_items):
    with col:
        st.markdown(
            f"""
            <div class="info-card" style="text-align:center; min-height:160px;">
                <span style="font-size:2rem;">{icon}</span>
                <h4 style="margin-top:8px;">{title}</h4>
                <p style="font-size:0.8rem;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.info("👈 Gunakan **sidebar** untuk berpindah halaman.", icon="💡")

# ─── Footer ─────────────────────────────────────────────────
footer()
