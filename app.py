# ============================================================
#  SISTEM KLASIFIKASI KOLEKTIBILITAS KREDIT — PT BPR MSA
#  Model utama: XGBoost  |  Pembanding: Random Forest, MLP
#  Jalankan dengan:  streamlit run app_skripsi.py
# ============================================================

import re
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
# 2. UTILITAS FORMAT RUPIAH
# ------------------------------------------------------------
def format_rupiah(angka):
    """Ubah angka menjadi teks Rupiah.
       1000000  ->  'Rp 1.000.000'   |   None  ->  '' (kosong)"""
    if angka is None:
        return ""
    return "Rp " + f"{angka:,.0f}".replace(",", ".")


def parse_rupiah(teks):
    """Ubah teks menjadi angka — semua karakter selain digit dibuang.
       'Rp 1.000.000' / '1.000.000' / '1000000'  ->  1000000.0
       Teks kosong atau tanpa angka  ->  None"""
    if teks is None:
        return None
    hanya_angka = re.sub(r"[^\d]", "", str(teks))
    if hanya_angka == "":
        return None
    return float(hanya_angka)


def rapikan_rupiah(kunci):
    """Callback: rapikan isi kolom menjadi 'Rp 1.000.000' secara otomatis.
       Dipanggil saat pengguna selesai mengetik (pindah kolom / tekan Enter)."""
    angka = parse_rupiah(st.session_state.get(kunci, ""))
    st.session_state[kunci] = format_rupiah(angka)   # None -> "" (kosong lagi)


def input_rupiah(label, kunci):
    """Kolom nominal Rupiah — awalnya kosong, otomatis diformat setelah diisi."""
    st.text_input(label, key=kunci, placeholder="Rp 0",
                  on_change=rapikan_rupiah, args=(kunci,))
    return parse_rupiah(st.session_state.get(kunci, ""))


# ------------------------------------------------------------
# 3. OPSI DROPDOWN (diambil dari dataset, otomatis tanpa duplikat)
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
# 4. FEATURE ENGINEERING (identik dengan training)
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
# 5. MUAT KETIGA MODEL (.pkl)
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
# 6. TAMPILAN
# ------------------------------------------------------------
st.set_page_config(page_title="Klasifikasi Kolektibilitas Kredit BPR MSA",
                   page_icon="🏦", layout="wide")

st.title("Sistem Klasifikasi Kolektibilitas Kredit")
st.caption("PT BPR Mahdani Sejahtera Abadi · Model utama: XGBoost · "
           "Hasil bersifat rekomendasi; keputusan akhir tetap pada analis kredit.")

models = muat_model()


# ---- Input dua kolom ----
kol1, kol2 = st.columns(2)

with kol1:
    st.subheader("Data Keuangan & Kredit")
    penghasilan_bulanan = input_rupiah("Penghasilan Bulanan", "in_penghasilan")
    angsuran_per_bulan  = input_rupiah("Angsuran per Bulan",  "in_angsuran")
    plafon_kredit       = input_rupiah("Plafon Kredit",       "in_plafon")
    sisa_pokok_pinjaman = input_rupiah("Sisa Pokok Pinjaman", "in_sisa_pokok")
    saldo_tabungan      = input_rupiah("Saldo Tabungan",      "in_saldo")
    nilai_jaminan       = input_rupiah("Nilai Jaminan",       "in_jaminan")
    tenor_bulan         = st.number_input("Tenor (bulan)", min_value=0, step=1,
                                          value=None, placeholder="contoh: 36")
    suku_bunga          = st.number_input("Suku Bunga (%)", min_value=0.0, step=0.5,
                                          value=None, placeholder="contoh: 12")

with kol2:
    st.subheader("Profil Nasabah & Lainnya")
    usia               = st.number_input("Usia (tahun)", min_value=17, step=1,
                                         value=None, placeholder="contoh: 35")
    jumlah_tanggunan   = st.number_input("Jumlah Tanggungan", min_value=0, step=1,
                                         value=None, placeholder="contoh: 2")
    lama_nasabah_bulan = st.number_input("Lama Menjadi Nasabah (bulan)", min_value=0, step=1,
                                         value=None, placeholder="contoh: 24")
    status_pernikahan  = st.selectbox("Status Pernikahan", OPSI["status_pernikahan"],
                                      index=None, placeholder="Pilih status pernikahan")
    pekerjaan_utama    = st.selectbox("Pekerjaan Utama", OPSI["pekerjaan_utama"],
                                      index=None, placeholder="Pilih pekerjaan utama")
    pekerjaan_detail   = st.selectbox("Pekerjaan Detail", OPSI["pekerjaan_detail"],
                                      index=None, placeholder="Pilih pekerjaan detail")
    produk_kredit      = st.selectbox("Produk Kredit", OPSI["produk_kredit"],
                                      index=None, placeholder="Pilih produk kredit")
    sumber_pembayaran  = st.selectbox("Sumber Pembayaran", OPSI["sumber_pembayaran"],
                                      index=None, placeholder="Pilih sumber pembayaran")
    kode_jenis_agunan  = st.selectbox("Kode Jenis Agunan", OPSI["kode_jenis_agunan"],
                                      index=None, placeholder="Pilih kode jenis agunan",
                                      format_func=lambda x: str(int(x)))
    kecamatan          = st.selectbox("Kecamatan", OPSI["kecamatan"],
                                      index=None, placeholder="Pilih kecamatan")
    pernah_restruktur  = st.selectbox("Pernah Restruktur?", ["Tidak", "Ya"],
                                      index=None, placeholder="Pilih Ya / Tidak")
    frekuensi_restrukturisasi = st.number_input("Frekuensi Restrukturisasi", min_value=0, step=1,
                                                value=None, placeholder="contoh: 0")

st.markdown("---")
submit = st.button("Prediksi Kolektibilitas", use_container_width=True, type="primary")


# ------------------------------------------------------------
# 7. PREDIKSI SAAT TOMBOL DITEKAN
# ------------------------------------------------------------
if submit:
    if MODEL_UTAMA not in models:
        st.error(f"Model utama ({MODEL_UTAMA}) belum termuat. Periksa berkas .pkl.")
        st.stop()

    # ---- Validasi: semua kolom wajib terisi ----
    wajib = {
        "Penghasilan Bulanan":       penghasilan_bulanan,
        "Angsuran per Bulan":        angsuran_per_bulan,
        "Plafon Kredit":             plafon_kredit,
        "Sisa Pokok Pinjaman":       sisa_pokok_pinjaman,
        "Saldo Tabungan":            saldo_tabungan,
        "Nilai Jaminan":             nilai_jaminan,
        "Tenor (bulan)":             tenor_bulan,
        "Suku Bunga":                suku_bunga,
        "Usia":                      usia,
        "Jumlah Tanggungan":         jumlah_tanggunan,
        "Lama Menjadi Nasabah":      lama_nasabah_bulan,
        "Status Pernikahan":         status_pernikahan,
        "Pekerjaan Utama":           pekerjaan_utama,
        "Pekerjaan Detail":          pekerjaan_detail,
        "Produk Kredit":             produk_kredit,
        "Sumber Pembayaran":         sumber_pembayaran,
        "Kode Jenis Agunan":         kode_jenis_agunan,
        "Kecamatan":                 kecamatan,
        "Pernah Restruktur":         pernah_restruktur,
        "Frekuensi Restrukturisasi": frekuensi_restrukturisasi,
    }
    kosong = [nama for nama, nilai in wajib.items() if nilai is None]
    if kosong:
        st.error("Kolom berikut belum diisi: " + ", ".join(kosong))
        st.stop()

    # ---- Validasi: penyebut tidak boleh nol (untuk hitung DSR & rasio tabungan) ----
    if penghasilan_bulanan <= 0:
        st.error("Penghasilan Bulanan harus lebih besar dari nol.")
        st.stop()
    if angsuran_per_bulan <= 0:
        st.error("Angsuran per Bulan harus lebih besar dari nol.")
        st.stop()

    # ---- Susun 1 baris data mentah (nama kolom harus sama dengan saat training) ----
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