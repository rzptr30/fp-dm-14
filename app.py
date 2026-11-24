"""
app.py - Aplikasi Streamlit Final Project Data Mining (Kelompok 14)
Klasifikasi Kategori ISPU Keesokan Hari di DKI Jakarta

Berkas yang dibaca (hasil dari skrip 02 dan 03, taruh satu folder dengan app.py):
  data_bersih.csv, hasil_model.csv, laporan_klasifikasi.csv,
  confusion_matrix.png, fitur_penting.png, model_terbaik.joblib

Jalankan lokal: streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ISPU DKI Jakarta - Final Project DM", layout="wide")

BASE = Path(__file__).parent
POLUTAN = ["pm_sepuluh", "pm_duakomalima", "sulfur_dioksida",
           "karbon_monoksida", "ozon", "nitrogen_dioksida"]
LABEL = {"pm_sepuluh": "PM10", "pm_duakomalima": "PM2.5", "sulfur_dioksida": "SO2",
         "karbon_monoksida": "CO", "ozon": "O3", "nitrogen_dioksida": "NO2"}
KELAS = ["BAIK", "SEDANG", "TIDAK SEHAT"]
WARNA = ["#4caf50", "#ffc107", "#f44336"]
KODE_STASIUN = {"Bundaran HI": "DKI1", "Kelapa Gading": "DKI2", "Jagakarsa": "DKI3",
                "Lubang Buaya": "DKI4", "Kebon Jeruk": "DKI5"}
NAMA_BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
              "Agustus", "September", "Oktober", "November", "Desember"]


# ---------------------------------------------------------------
# Pemuatan data
# ---------------------------------------------------------------
@st.cache_data
def muat_data() -> pd.DataFrame:
    return pd.read_csv(BASE / "data_bersih.csv", parse_dates=["tanggal_lengkap"])


@st.cache_data
def muat_hasil():
    hasil = pd.read_csv(BASE / "hasil_model.csv")
    laporan = pd.read_csv(BASE / "laporan_klasifikasi.csv", index_col=0)
    return hasil, laporan


@st.cache_resource
def muat_model():
    return joblib.load(BASE / "model_terbaik.joblib")


def wajib_ada(*nama_berkas: str) -> None:
    hilang = [n for n in nama_berkas if not (BASE / n).exists()]
    if hilang:
        st.error(f"Berkas belum ada di folder aplikasi: {', '.join(hilang)}")
        st.stop()


def tampil(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------
# Halaman 1: Ringkasan
# ---------------------------------------------------------------
def halaman_ringkasan(df: pd.DataFrame) -> None:
    st.title("Prediksi Kategori Kualitas Udara (ISPU) Keesokan Hari di DKI Jakarta")
    st.caption("Final Project Data Mining - Kelompok 14 - Task: Klasifikasi - Kerangka kerja: CRISP-DM")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baris data bersih", f"{len(df):,}".replace(",", "."))
    k2.metric("Stasiun pemantau", df["stasiun"].nunique())
    k3.metric("Awal data", df["tanggal_lengkap"].min().strftime("%d %b %Y"))
    k4.metric("Akhir data", df["tanggal_lengkap"].max().strftime("%d %b %Y"))

    st.subheader("Latar belakang singkat")
    st.write(
        "ISPU (Indeks Standar Pencemar Udara) menggambarkan mutu udara dalam kategori Baik, Sedang, "
        "Tidak Sehat, Sangat Tidak Sehat, dan Berbahaya. Kategori dihitung langsung dari nilai tertinggi "
        "sub-indeks polutan, sehingga memprediksi kategori hari yang sama dari polutan hari itu hanya "
        "menebak ulang rumus. Karena itu proyek ini memprediksi kategori **hari berikutnya** dari kondisi "
        "polutan hari ini dan kemarin."
    )

    st.subheader("Alur pengerjaan (CRISP-DM)")
    st.markdown(
        "1. **Business Understanding**: rumusan masalah dan tujuan\n"
        "2. **Data Understanding**: profil data dan EDA (halaman *EDA*)\n"
        "3. **Data Preparation**: pembersihan format, fitur lag per stasiun, split berdasarkan waktu\n"
        "4. **Modeling**: Decision Tree, Random Forest, dan KNN dengan tuning berbasis waktu\n"
        "5. **Evaluation**: macro-F1 dibanding baseline (halaman *Hasil Model*)\n"
        "6. **Deployment**: aplikasi ini (halaman *Coba Prediksi*)"
    )

    st.subheader("Sumber data")
    st.write(
        "Data Indeks Standar Pencemar Udara (ISPU) di Provinsi DKI Jakarta, Dinas Lingkungan Hidup "
        "Provinsi DKI Jakarta, melalui portal Satu Data Jakarta (satudata.jakarta.go.id). "
        "Periode 1 Januari 2024 sampai 30 November 2025, lima stasiun pemantau."
    )


# ---------------------------------------------------------------
# Halaman 2: EDA
# ---------------------------------------------------------------
def halaman_eda(df: pd.DataFrame) -> None:
    st.title("Exploratory Data Analysis")

    st.sidebar.subheader("Filter EDA")
    pilih_stasiun = st.sidebar.multiselect(
        "Stasiun", sorted(df["stasiun"].unique()), default=sorted(df["stasiun"].unique()))
    tmin, tmax = df["tanggal_lengkap"].min().date(), df["tanggal_lengkap"].max().date()
    rentang = st.sidebar.date_input("Rentang tanggal", (tmin, tmax), min_value=tmin, max_value=tmax)

    if len(rentang) != 2 or not pilih_stasiun:
        st.info("Pilih minimal satu stasiun dan rentang tanggal lengkap (awal dan akhir).")
        return
    d = df[df["stasiun"].isin(pilih_stasiun)
           & (df["tanggal_lengkap"].dt.date >= rentang[0])
           & (df["tanggal_lengkap"].dt.date <= rentang[1])]
    if d.empty:
        st.warning("Tidak ada data pada filter ini.")
        return
    st.caption(f"{len(d):,} baris data".replace(",", ".") + " sesuai filter.")

    tab1, tab2, tab3, tab4 = st.tabs(["Kategori", "Pola musiman", "Polutan", "Peluang transisi"])

    with tab1:
        kiri, kanan = st.columns(2)
        hitung = d["kategori"].value_counts().reindex(KELAS).fillna(0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(KELAS, hitung.values, color=WARNA)
        ax.set_ylabel("Jumlah hari-stasiun")
        ax.set_title("Distribusi kategori ISPU")
        kiri.pyplot(fig)
        plt.close(fig)

        per = pd.crosstab(d["stasiun"], d["kategori"], normalize="index").reindex(columns=KELAS).fillna(0) * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        per.plot.barh(stacked=True, color=WARNA, ax=ax)
        ax.set_xlabel("%")
        ax.set_title("Komposisi kategori per stasiun")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        kanan.pyplot(fig)
        plt.close(fig)
        st.write("Kelas sangat timpang: kategori Sedang mendominasi. Karena itu metrik utama adalah macro-F1, bukan accuracy.")

    with tab2:
        bulanan = d["kategori"].eq("TIDAK SEHAT").groupby(d["tanggal_lengkap"].dt.to_period("M")).mean() * 100
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(bulanan.index.astype(str), bulanan.values, color="#f44336")
        ax.set_ylabel("% hari Tidak Sehat")
        ax.set_title("Persentase hari TIDAK SEHAT per bulan")
        plt.setp(ax.get_xticklabels(), rotation=60)
        tampil(fig)
        st.write("Hari Tidak Sehat menumpuk pada pertengahan tahun, sehingga bulan menjadi fitur yang relevan "
                 "dan data uji (yang memuat puncak Juni-Juli 2025) memiliki proporsi Tidak Sehat lebih tinggi.")

    with tab3:
        kiri, kanan = st.columns(2)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.boxplot([d[k].dropna() for k in POLUTAN])
        ax.set_xticklabels([LABEL[k] for k in POLUTAN])
        ax.set_title("Sebaran sub-indeks polutan")
        kiri.pyplot(fig)
        plt.close(fig)

        kosong = d[POLUTAN].isna().sum()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar([LABEL[k] for k in POLUTAN], kosong.values, color="#607d8b")
        ax.set_title("Nilai kosong per polutan (sebelum imputasi)")
        kanan.pyplot(fig)
        plt.close(fig)

        kiri, kanan = st.columns(2)
        kor = d[POLUTAN].corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        gambar = ax.imshow(kor, cmap="coolwarm", vmin=-1, vmax=1)
        nama = [LABEL[k] for k in POLUTAN]
        ax.set_xticks(range(6), nama, rotation=45)
        ax.set_yticks(range(6), nama)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{kor.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(gambar)
        ax.set_title("Korelasi antar polutan")
        kiri.pyplot(fig)
        plt.close(fig)

        kritis = d["parameter_pencemar_kritis"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(kritis.index.astype(str), kritis.values, color="#3f51b5")
        ax.set_title("Polutan kritis (penentu kategori)")
        kanan.pyplot(fig)
        plt.close(fig)

    with tab4:
        pasangan = []
        for _, g in d.groupby("kode_stasiun"):
            g = g.set_index("tanggal_lengkap").sort_index()
            g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
            pasangan.append(pd.DataFrame({"hari_ini": g["kategori"], "besok": g["kategori"].shift(-1)}))
        pasangan = pd.concat(pasangan, ignore_index=True).dropna()
        if pasangan.empty:
            st.info("Data pada filter terlalu sedikit untuk menghitung transisi.")
        else:
            tr = (pd.crosstab(pasangan["hari_ini"], pasangan["besok"], normalize="index")
                  .reindex(index=KELAS, columns=KELAS).fillna(0) * 100)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            gambar = ax.imshow(tr, cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks(range(3), KELAS)
            ax.set_yticks(range(3), KELAS)
            ax.set_xlabel("Kategori besok")
            ax.set_ylabel("Kategori hari ini")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{tr.iloc[i, j]:.0f}%", ha="center", va="center",
                            color="white" if tr.iloc[i, j] > 50 else "black")
            fig.colorbar(gambar)
            ax.set_title("Peluang kategori besok, diketahui kategori hari ini")
            tampil(fig)
            st.write("Kategori cenderung bertahan dari hari ke hari, sehingga tebakan \"besok sama dengan hari ini\" "
                     "menjadi pembanding (baseline) yang kuat. Namun hari Tidak Sehat sering kembali ke Sedang di hari berikutnya.")


# ---------------------------------------------------------------
# Halaman 3: Hasil model
# ---------------------------------------------------------------
def halaman_hasil() -> None:
    st.title("Hasil Modeling dan Evaluasi")
    wajib_ada("hasil_model.csv", "laporan_klasifikasi.csv")
    hasil, laporan = muat_hasil()

    st.subheader("Perbandingan pada data uji")
    st.dataframe(hasil, hide_index=True)
    st.caption("Data uji: 1 Juni 2025 s.d. akhir data. Data latih: sebelum itu (split berdasarkan waktu). "
               "Macro-F1 dipilih karena kelas sangat timpang.")

    model_saja = hasil[~hasil["Model"].str.startswith("Baseline")]
    baseline = hasil[hasil["Model"] == "Baseline: besok = hari ini"]
    if not model_saja.empty and not baseline.empty:
        terbaik = model_saja.sort_values("Macro-F1", ascending=False).iloc[0]
        f1_base = float(baseline["Macro-F1"].iloc[0])
        if terbaik["Macro-F1"] > f1_base:
            st.success(f"Model terbaik: {terbaik['Model']} (macro-F1 {terbaik['Macro-F1']:.3f}), "
                       f"melampaui baseline \"besok = hari ini\" ({f1_base:.3f}).")
        else:
            st.warning(f"Model terbaik: {terbaik['Model']} (macro-F1 {terbaik['Macro-F1']:.3f}) belum "
                       f"melampaui baseline \"besok = hari ini\" ({f1_base:.3f}). "
                       "Lihat pembahasan di laporan mengenai kekuatan baseline dan nilai model pada kelas Tidak Sehat.")

    st.subheader("Laporan klasifikasi model terbaik")
    st.dataframe(laporan)

    kiri, kanan = st.columns(2)
    if (BASE / "confusion_matrix.png").exists():
        kiri.image(str(BASE / "confusion_matrix.png"), caption="Confusion matrix model terbaik")
    if (BASE / "fitur_penting.png").exists():
        kanan.image(str(BASE / "fitur_penting.png"), caption="Fitur terpenting (Random Forest)")


# ---------------------------------------------------------------
# Halaman 4: Coba prediksi
# ---------------------------------------------------------------
def halaman_prediksi(df: pd.DataFrame) -> None:
    st.title("Coba Prediksi Kategori Besok")
    wajib_ada("model_terbaik.joblib")
    model = muat_model()
    st.write("Masukkan kondisi polutan hari ini dan kemarin, lalu model memperkirakan kategori ISPU besok. "
             "Ini demonstrasi model; akurasinya terbatas (lihat halaman Hasil Model).")

    median = df[POLUTAN].median()
    c1, c2, c3 = st.columns(3)
    stasiun = c1.selectbox("Stasiun", list(KODE_STASIUN))
    bulan_nama = c2.selectbox("Bulan hari ini", NAMA_BULAN, index=5)

    kiri, kanan = st.columns(2)
    kiri.markdown("**Hari ini (sub-indeks)**")
    kanan.markdown("**Kemarin (sub-indeks)**")
    hari_ini, kemarin = {}, {}
    for k in POLUTAN:
        hari_ini[k] = kiri.slider(LABEL[k] + " hari ini", 0, 200, int(median[k]), key=f"t_{k}")
        kemarin[k] = kanan.slider(LABEL[k] + " kemarin", 0, 200, int(median[k]), key=f"k_{k}")

    baris = {"kode_stasiun": KODE_STASIUN[stasiun], "bulan": NAMA_BULAN.index(bulan_nama) + 1}
    for k in POLUTAN:
        baris[k] = hari_ini[k]
        baris[k + "_kemarin"] = kemarin[k]
    X = pd.DataFrame([baris])

    if st.button("Prediksi", type="primary"):
        prediksi = model.predict(X)[0]
        st.subheader(f"Prediksi kategori besok: {prediksi}")
        if hasattr(model, "predict_proba"):
            peluang = pd.Series(model.predict_proba(X)[0], index=model.classes_).reindex(KELAS).fillna(0)
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh(KELAS, peluang.values * 100, color=WARNA)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Peluang (%)")
            tampil(fig)


# ---------------------------------------------------------------
# Navigasi
# ---------------------------------------------------------------
wajib_ada("data_bersih.csv")
data = muat_data()

halaman = st.sidebar.radio("Halaman", ["Ringkasan", "EDA", "Hasil Model", "Coba Prediksi"])
if halaman == "Ringkasan":
    halaman_ringkasan(data)
elif halaman == "EDA":
    halaman_eda(data)
elif halaman == "Hasil Model":
    halaman_hasil()
else:
    halaman_prediksi(data)
