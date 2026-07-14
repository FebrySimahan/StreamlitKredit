# ============================================================
#  SISTEM KLASIFIKASI KOLEKTIBILITAS KREDIT — PT BPR MSA
#  Model utama: XGBoost  |  Pembanding: Random Forest, MLP
#  Jalankan dengan:  streamlit run app.py
# ============================================================

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------
# 1. DEFINISI CUSTOM TRANSFORMER
#    WAJIB ada & PERSIS SAMA dengan yang dipakai saat training,
#    karena pickle membutuhkan definisi class ini untuk membuka
#    berkas .pkl. (disalin dari skrip pelatihan)
# ------------------------------------------------------------
class TargetEncoderSmooth(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=10):
        self.cols = cols; self.smoothing = smoothing
    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_ = y.mean(); self.maps_ = {}
        for c in self.cols:
            s = y.groupby(X[c]).agg(['mean', 'count'])
            self.maps_[c] = ((s['count'] * s['mean'] + self.smoothing * self.global_)
                             / (s['count'] + self.smoothing)).to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c + '_te'] = X[c].map(self.maps_[c]).fillna(self.global_)
            X = X.drop(columns=c)
        return X


class GeoRiskScore(BaseEstimator, TransformerMixin):
    def __init__(self, col='kecamatan', smoothing=10):
        self.col = col; self.smoothing = smoothing
    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_ = y.mean()
        s = y.groupby(X[self.col]).agg(['mean', 'count'])
        self.map_ = (((s['count'] * s['mean'] + self.smoothing * self.global_)
                      / (s['count'] + self.smoothing)) * 100).to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        X['GRS'] = X[self.col].map(self.map_).fillna(self.global_ * 100)
        return X.drop(columns=self.col)


# --- Agar pickle menemukan class di namespace __main__ (mencegah error saat load) ---
import __main__
__main__.TargetEncoderSmooth = TargetEncoderSmooth
__main__.GeoRiskScore = GeoRiskScore


# ------------------------------------------------------------
# 2. OPSI DROPDOWN (diambil dari dataset, otomatis tanpa duplikat)
# ------------------------------------------------------------
@st.cache_data
def muat_opsi():
    df = pd.read_csv("DATA_LENGKAP.csv")         # dataset sumber (sama dgn training)

    def opsi(kolom):
        # dropna() buang kosong, unique() buang duplikat, sorted() urutkan
        return sorted(df[kolom].dropna().unique().tolist())

    return {
        "status_pernikahan": opsi("status_pernikahan"),
        "pekerjaan_utama":   opsi("pekerjaan_utama"),
        "pekerjaan_detail":  opsi("pekerjaan_detail"),
        "produk_kredit":     opsi("produk_kredit"),
        "sumber_pembayaran": opsi("sumber_pembayaran"),
        "kode_jenis_agunan": sorted(df["kode_jenis_agunan"].dropna().unique().tolist()),
        "kecamatan":         opsi("kecamatan"),
    }

OPSI = muat_opsi()
# ------------------------------------------------------------
# 3. FEATURE ENGINEERING (identik dengan training)
#    Menghitung DSR, LTV, rasio_tabungan, dan konversi restrukturisasi.
# ------------------------------------------------------------
def buat_fitur(d):
    d = d.copy()
    d['DSR'] = (d['angsuran_per_bulan'] / d['penghasilan_bulanan'] * 100).clip(0, 200)
    jaminan = d['nilai_jaminan'].replace(0, np.nan)
    d['LTV'] = (d['sisa_pokok_pinjaman'] / jaminan * 100).clip(0, 300).fillna(300)
    d['rasio_tabungan'] = (d['saldo_tabungan'] / d['angsuran_per_bulan']).clip(0, 100)
    d['pernah_restruktur'] = (d['pernah_restruktur'].astype(str).str.strip() == 'Ya').astype(int)
    return d


# ------------------------------------------------------------
# 4. MUAT KETIGA MODEL (.pkl)
#    Sesuaikan nama berkas dengan yang Anda simpan saat training.
# ------------------------------------------------------------
MODEL_FILES = {
    "XGBoost":       "xgbfinal.pkl",
    "Random Forest": "rffinal.pkl",
    "MLP":           "mlpfinal.pkl",
}
MODEL_UTAMA = "XGBoost"   # model terbaik hasil penelitian


@st.cache_resource
def muat_model():
    model = {}
    for nama, path in MODEL_FILES.items():
        try:
            with open(path, "rb") as f:
                model[nama] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Berkas model '{path}' tidak ditemukan. Lewati {nama}.")
    return model


# ------------------------------------------------------------
# 5. TAMPILAN
# ------------------------------------------------------------
st.set_page_config(page_title="Klasifikasi Kolektibilitas Kredit BPR MSA",
                   page_icon="🏦", layout="wide")

st.title("Sistem Klasifikasi Kolektibilitas Kredit")
st.caption("PT BPR Mahdani Sejahtera Abadi · Model utama: XGBoost · "
           "Hasil bersifat rekomendasi; keputusan akhir tetap pada analis kredit.")

models = muat_model()

# ---- Form input dua kolom ----
with st.form("form_nasabah"):
    kol1, kol2 = st.columns(2)

    with kol1:
        st.subheader("Data Keuangan & Kredit")
        penghasilan_bulanan = st.number_input("Penghasilan Bulanan (Rp)", min_value=0.0, value=5_000_000.0, step=500_000.0)
        angsuran_per_bulan  = st.number_input("Angsuran per Bulan (Rp)",   min_value=0.0, value=1_500_000.0, step=100_000.0)
        plafon_kredit       = st.number_input("Plafon Kredit (Rp)",        min_value=0.0, value=50_000_000.0, step=1_000_000.0)
        sisa_pokok_pinjaman = st.number_input("Sisa Pokok Pinjaman (Rp)",  min_value=0.0, value=30_000_000.0, step=1_000_000.0)
        saldo_tabungan      = st.number_input("Saldo Tabungan (Rp)",       min_value=0.0, value=2_000_000.0, step=100_000.0)
        nilai_jaminan       = st.number_input("Nilai Jaminan (Rp)",        min_value=0.0, value=60_000_000.0, step=1_000_000.0)
        tenor_bulan         = st.number_input("Tenor (bulan)",             min_value=0,   value=36, step=1)
        suku_bunga          = st.number_input("Suku Bunga (%)",            min_value=0.0, value=12.0, step=0.5)

    with kol2:
        st.subheader("Profil Nasabah & Lainnya")
        usia               = st.number_input("Usia (tahun)",            min_value=17, value=35, step=1)
        jumlah_tanggunan   = st.number_input("Jumlah Tanggungan",       min_value=0,  value=2, step=1)
        lama_nasabah_bulan = st.number_input("Lama Menjadi Nasabah (bulan)", min_value=0, value=24, step=1)
        status_pernikahan  = st.selectbox("Status Pernikahan", OPSI["status_pernikahan"])
        pekerjaan_utama    = st.selectbox("Pekerjaan Utama",   OPSI["pekerjaan_utama"])
        pekerjaan_detail   = st.selectbox("Pekerjaan Detail",  OPSI["pekerjaan_detail"])
        produk_kredit      = st.selectbox("Produk Kredit",     OPSI["produk_kredit"])
        sumber_pembayaran  = st.selectbox("Sumber Pembayaran", OPSI["sumber_pembayaran"])
        kode_jenis_agunan  = st.selectbox("Kode Jenis Agunan", OPSI["kode_jenis_agunan"],
                                        format_func=lambda x: str(int(x)))
        kecamatan          = st.selectbox("Kecamatan",         OPSI["kecamatan"])
        pernah_restruktur  = st.selectbox("Pernah Restruktur?", ["Tidak", "Ya"])
        frekuensi_restrukturisasi = st.number_input("Frekuensi Restrukturisasi", min_value=0, value=0, step=1)

    submit = st.form_submit_button("Prediksi Kolektibilitas", use_container_width=True, type="primary")


# ------------------------------------------------------------
# 6. PREDIKSI SAAT TOMBOL DITEKAN
# ------------------------------------------------------------
if submit:
    if MODEL_UTAMA not in models:
        st.error(f"Model utama ({MODEL_UTAMA}) belum termuat. Periksa berkas .pkl.")
        st.stop()

    # susun 1 baris data mentah (nama kolom harus sama dengan saat training)
    data = pd.DataFrame([{
        "usia": usia,
        "status_pernikahan": status_pernikahan,
        "jumlah_tanggunan": jumlah_tanggunan,
        "pekerjaan_utama": pekerjaan_utama,
        "pekerjaan_detail": pekerjaan_detail,
        "sumber_pembayaran": sumber_pembayaran,
        "penghasilan_bulanan": penghasilan_bulanan,
        "produk_kredit": produk_kredit,
        "plafon_kredit": plafon_kredit,
        "sisa_pokok_pinjaman": sisa_pokok_pinjaman,
        "angsuran_per_bulan": angsuran_per_bulan,
        "tenor_bulan": tenor_bulan,
        "suku_bunga": suku_bunga,
        "pernah_restruktur": pernah_restruktur,
        "frekuensi_restrukturisasi": frekuensi_restrukturisasi,
        "saldo_tabungan": saldo_tabungan,
        "lama_nasabah_bulan": lama_nasabah_bulan,
        "nilai_jaminan": nilai_jaminan,
        "kode_jenis_agunan": kode_jenis_agunan,
        "kecamatan": kecamatan,
    }])

    # feature engineering (DSR, LTV, rasio_tabungan, konversi restrukturisasi)
    data = buat_fitur(data)

    LABEL = {0: "LANCAR", 1: "TIDAK LANCAR"}

    def prediksi(model):
        pred = int(model.predict(data)[0])
        proba_tl = float(model.predict_proba(data)[0][1])  # peluang Tidak Lancar
        return pred, proba_tl

    # ---- Hasil model utama (XGBoost) ----
    st.markdown("---")
    st.subheader("Hasil Prediksi (Model Utama: XGBoost)")
    pred_u, proba_u = prediksi(models[MODEL_UTAMA])

    if pred_u == 0:
        st.success(f"Status Kolektibilitas: **{LABEL[pred_u]}**")
    else:
        st.error(f"Status Kolektibilitas: **{LABEL[pred_u]}**")

    c1, c2 = st.columns(2)
    c1.metric("Peluang Tidak Lancar", f"{proba_u*100:.1f}%")
    c2.metric("Peluang Lancar", f"{(1-proba_u)*100:.1f}%")
    st.progress(proba_u, text=f"Tingkat risiko (peluang Tidak Lancar): {proba_u*100:.1f}%")

    # ---- Perbandingan dengan model lain ----
    st.markdown("---")
    st.subheader("Perbandingan Antar Model")
    st.caption("XGBoost adalah model terbaik pada penelitian ini. Prediksi Random Forest "
               "dan MLP ditampilkan sebagai pembanding.")

    urutan = [m for m in ["XGBoost", "Random Forest", "MLP"] if m in models]
    kolom = st.columns(len(urutan))
    baris_tabel = []

    for kol, nama in zip(kolom, urutan):
        pred, proba_tl = prediksi(models[nama])
        with kol:
            st.markdown(f"**{nama}**" + ("  ·  _model utama_" if nama == MODEL_UTAMA else ""))
            if pred == 0:
                st.success(LABEL[pred])
            else:
                st.error(LABEL[pred])
            st.metric("Peluang Tidak Lancar", f"{proba_tl*100:.1f}%")
        baris_tabel.append({
            "Model": nama + (" (utama)" if nama == MODEL_UTAMA else ""),
            "Prediksi": LABEL[pred],
            "Peluang Tidak Lancar": f"{proba_tl*100:.1f}%",
            "Peluang Lancar": f"{(1-proba_tl)*100:.1f}%",
        })

    st.markdown("##### Ringkasan")
    st.dataframe(pd.DataFrame(baris_tabel), hide_index=True, use_container_width=True)

    # ---- Catatan kesepakatan model ----
    prediksi_semua = [prediksi(models[m])[0] for m in urutan]
    if len(set(prediksi_semua)) == 1:
        st.info("Seluruh model memberikan prediksi yang sama.")
    else:
        st.warning("Terdapat perbedaan prediksi antar model. Disarankan meninjau kembali "
                   "profil nasabah dan mengutamakan hasil model utama (XGBoost).")
