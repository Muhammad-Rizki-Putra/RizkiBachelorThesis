"""
Tentang Page
"""
import streamlit as st
from utils import inject_custom_css, section_header, divider, render_sidebar

st.set_page_config(page_title="Tentang", page_icon="ℹ️", layout="wide")
inject_custom_css()
render_sidebar()

st.markdown('<div class="hero-title">ℹ️ Tentang Penelitian</div>', unsafe_allow_html=True)
divider()

col1, col2 = st.columns([2, 1])

with col1:
    section_header("Abstrak Penelitian", "📝")
    st.markdown(
        """
        Disleksia merupakan gangguan belajar spesifik pada kemampuan membaca dan mengeja, 
        yang berakar pada neurologi dan diperlihatkan pada simptom defisit pada kemampuan fonologi.
        
        Skripsi ini mengeksplorasi potensi penggunaan **teknologi eye-tracking** dipadukan dengan 
        teknik **machine learning** dan **deep learning** untuk mendeteksi disleksia. Dengan menganalisis 
        pola pergerakan mata—seperti fiksasi, saccade, dan regresi—saat subjek 
        membaca berbagai jenis teks, kita dapat mengidentifikasi biomarker yang terkait dengan perilaku membaca penderita disleksia.
        
        Penelitian ini membandingkan model machine learning tabular tradisional (mengekstraksi fitur 
        statistik dari pergerakan mata) dengan model citra terbaru (mengubah jalur pemindaian menjadi 
        representasi visual untuk CNN).
        """
    )
    
    section_header("Dataset", "📁")
    st.markdown(
        """
        Dataset ini terdiri dari rekaman eye-tracking frekuensi tinggi dari **70 subjek** 
        (35 didiagnosis disleksia, 35 subjek kontrol).
        
        Tiga tugas membaca diberikan:
        - **Tugas 1 (Suku Kata):** Membaca suku kata terpisah.
        - **Tugas 4 (Teks Bermakna):** Membaca paragraf koheren.
        - **Tugas 5 (Teks Semu):** Membaca paragraf kata non-baku (pseudo-word) untuk menguji kemampuan pemecahan fonetik.
        """
    )

with col2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 👨‍🎓 Penulis")
    st.markdown("[**Muhammad Rizki Putra**](https://www.linkedin.com/in/rizki-putra/)")
    st.markdown("*Skripsi Sarjana 2026*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 👥 Dosen Pembimbing")
    st.markdown("- [**Dr. Mira Suryani, S.Pd., M.Kom.**](https://scholar.google.com/citations?user=EBCeq7gAAAAJ&hl=en)")
    st.markdown("- [**Erick Paulus, S.Si., M.Kom.**](https://scholar.google.com/citations?user=MyTaEvsAAAAJ&hl=id)")
    st.markdown("- [**Dr. rer. nat. Shally Novita**](https://scholar.google.com/citations?user=FK8FJegAAAAJ&hl=en)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Teknologi yang Digunakan")
    st.markdown(
        """
        - **UI/UX:** Streamlit
        - **Pemrosesan Data:** Pandas, NumPy
        - **Visualisasi:** Plotly
        - **Machine Learning:** Scikit-Learn, XGBoost, TensorFlow/Keras
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

divider()
st.markdown("<p style='text-align:center; color:#5A6473; font-size:0.9rem;'>Universitas Padjadjaran | 2026</p>", unsafe_allow_html=True)
