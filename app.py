# ==============================================================================
# 1. IMPORT LIBRARY
# ==============================================================================
# Penjelasan:
# Bagian ini adalah langkah pertama di mana kita memuat semua "alat" atau
# pustaka (library) yang dibutuhkan oleh aplikasi kita.
# Setiap library memiliki fungsi spesifik.

import streamlit as st  # Untuk membuat antarmuka web interaktif (dashboard).
import pandas as pd  # Untuk memanipulasi data dalam bentuk tabel (DataFrame).
import numpy as np  # Untuk operasi numerik, terutama dalam perhitungan matematika.
import lightgbm as lgb  # Ini adalah library untuk model machine learning utama kita, LightGBM.
import matplotlib.pyplot as plt  # Untuk membuat grafik dan plot statis.
import seaborn as sns  # Untuk membuat visualisasi data yang lebih menarik secara estetika.
from sklearn.model_selection import train_test_split, GridSearchCV  # Untuk membagi data dan mencari hyperparameter terbaik.
from sklearn.preprocessing import StandardScaler  # Untuk menormalisasi data (menyamakan skala fitur).
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error  # Untuk mengukur performa model.

# ==============================================================================
# 2. KONFIGURASI HALAMAN APLIKASI
# ==============================================================================
# Penjelasan:
# Di sini kita mengatur tampilan dasar dari halaman web aplikasi kita.

st.set_page_config(layout="wide")  # Membuat layout halaman menjadi lebar agar konten tidak terlalu sempit.
st.title("Prediksi Harga Cardano (ADA) dengan LightGBM & Analisis Overfitting")
st.write("Aplikasi ini melatih model LightGBM untuk memprediksi harga Cardano dan menganalisis kinerjanya.")

# ==============================================================================
# 3. FUNGSI-FUNGSI UTAMA
# ==============================================================================
# Penjelasan:
# Kita mendefinisikan fungsi-fungsi utama di sini agar kode lebih rapi dan
# bisa digunakan kembali. Penggunaan @st.cache_data dan @st.cache_resource
# adalah teknik optimasi agar aplikasi berjalan cepat.

# Dekorator @st.cache_data digunakan untuk fungsi yang memproses data.
# Streamlit akan menyimpan hasil dari fungsi ini di cache. Jika fungsi
# dipanggil lagi dengan input yang sama, Streamlit akan langsung
# memberikan hasil yang tersimpan tanpa menjalankan ulang seluruh proses.
# Ini sangat menghemat waktu, terutama untuk proses pemuatan data yang berat.
@st.cache_data
def load_and_process_data(path):
    """
    Fungsi ini bertanggung jawab untuk 3 hal:
    1. Memuat dataset dari file .csv.
    2. Membersihkan dan mengubah format data (pra-pemrosesan).
    3. Membuat fitur baru dari data yang ada (feature engineering).
    """
    st.info("Memuat dan memproses data untuk pertama kali...")
    df_raw = pd.read_csv(path)  # Membaca file CSV menjadi DataFrame pandas.
    df = df_raw.copy()  # Membuat salinan agar data asli tidak berubah.

    # --- Pra-pemrosesan Data ---
    df['Date'] = pd.to_datetime(df['Date'])  # Mengubah kolom 'Date' dari teks menjadi format tanggal.
    df = df.sort_values(by="Date").reset_index(drop=True)  # Mengurutkan data berdasarkan tanggal.

    # Fungsi kecil untuk mengubah volume dari format teks (misal, "519.20M") menjadi angka.
    def convert_to_numeric(volume_str):
        if isinstance(volume_str, str):
            volume_str = volume_str.replace(',', '').strip().upper()
            if 'M' in volume_str: return float(volume_str.replace('M', '')) * 1_000_000
            if 'K' in volume_str: return float(volume_str.replace('K', '')) * 1_000
            if 'B' in volume_str: return float(volume_str.replace('B', '')) * 1_000_000_000
            try: return float(volume_str)
            except ValueError: return np.nan
        return volume_str

    df['Vol.'] = df['Vol.'].apply(convert_to_numeric)  # Menerapkan fungsi konversi ke kolom volume.
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Memastikan semua kolom harga adalah angka.
    
    df.rename(columns={'Vol.': 'Volume'}, inplace=True)  # Mengganti nama kolom agar lebih jelas.

    # --- Feature Engineering ---
    # Proses ini adalah inti dari bagaimana model kita "melihat" masa lalu.
    # Kita membuat fitur baru yang merupakan nilai dari hari-hari sebelumnya (lag).
    lags = [1, 7, 30]  # Kita ingin melihat data dari 1 hari, 7 hari, dan 30 hari yang lalu.
    columns_to_lag = ['Price', 'Open', 'High', 'Low', 'Volume']
    
    # Loop untuk membuat kolom-kolom lag.
    # Contoh: df['Price_Lag1'] akan berisi nilai 'Price' dari baris sebelumnya.
    for col in columns_to_lag:
        for lag in lags:
            df[f'{col}_Lag{lag}'] = df[col].shift(lag)
    
    # Setelah membuat fitur lag, baris-baris pertama akan memiliki nilai kosong (NaN)
    # karena tidak ada data masa lalu yang cukup. Baris-baris ini harus dihapus.
    df.dropna(inplace=True)
    
    return df, df_raw  # Mengembalikan data yang sudah diproses dan data asli.

# Dekorator @st.cache_resource digunakan untuk menyimpan objek yang "berat"
# seperti model machine learning atau koneksi database.
@st.cache_resource
def run_grid_search(X_train, y_train, param_grid):
    """
    Fungsi ini menjalankan GridSearchCV untuk mencari kombinasi hyperparameter
    terbaik secara otomatis. Proses ini sangat intensif secara komputasi.
    """
    with st.spinner("⏳ Menjalankan GridSearchCV... Ini mungkin butuh beberapa waktu."):
        # Inisialisasi GridSearchCV
        grid_search = GridSearchCV(
            estimator=lgb.LGBMRegressor(random_state=42),  # Model dasar yang akan diuji.
            param_grid=param_grid,  # "Kamus" berisi semua hyperparameter yang ingin dicoba.
            cv=5,  # Cross-validation 5-fold, untuk evaluasi yang lebih stabil.
            scoring='neg_mean_squared_error',  # Metrik untuk menilai kombinasi terbaik.
            n_jobs=-1  # Menggunakan semua core CPU agar proses lebih cepat.
        )
        grid_search.fit(X_train, y_train)  # Memulai proses pencarian.
        return grid_search.best_params_  # Mengembalikan kombinasi parameter terbaik.

@st.cache_resource
def train_final_model(params, X_train, y_train, X_val, y_val):
    """
    Fungsi ini melatih model LightGBM final dengan menggunakan hyperparameter
    terbaik yang telah ditemukan.
    """
    st.info("Melatih model final dengan parameter terbaik...")
    final_model = lgb.LGBMRegressor(**params, random_state=42)  # Inisialisasi model dengan parameter terbaik.
    
    # Memulai proses pelatihan.
    final_model.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],  # Data validasi untuk memantau performa.
                    eval_metric='rmse',  # Metrik yang dipantau.
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])  # Menerapkan Early Stopping.
    return final_model

# ==============================================================================
# 4. ALUR KERJA UTAMA APLIKASI
# ==============================================================================
# Penjelasan:
# Ini adalah bagian utama di mana semua fungsi di atas dipanggil secara
# berurutan untuk menjalankan aplikasi.

# --- Langkah 1: Memuat dan Memproses Data ---
file_path = "dataset.csv"
try:
    df, df_original = load_and_process_data(file_path)
    st.success(f"Dataset '{file_path}' berhasil dimuat dan diproses.")
except FileNotFoundError:
    st.error(f"Error: File '{file_path}' tidak ditemukan. Pastikan nama file sudah benar.")
    st.stop()

# --- Tampilan Data (Opsional) ---
with st.expander("📊 Tampilkan Dataset Asli Cardano (ADA)"):
    st.dataframe(df_original)

# --- Langkah 2: Memisahkan Fitur (X) dan Target (y) ---
# Fitur (X) adalah semua informasi yang kita gunakan untuk membuat prediksi (semua kolom Lag).
# Target (y) adalah apa yang ingin kita prediksi (kolom 'Price').
feature_columns = [col for col in df.columns if '_Lag' in col]
X = df[feature_columns]
y = df[['Price']] # Menggunakan kurung siku ganda agar y tetap menjadi DataFrame.

# --- Langkah 3: Pengaturan dan Pembagian Data ---
st.sidebar.title("Pengaturan Model")
# Slider ini memungkinkan pengguna memilih rasio pembagian data secara interaktif.
test_size_ratio = st.sidebar.slider("Pilih Rasio Data Uji", 0.1, 0.4, 0.2, 0.05)
validation_size_ratio = 0.2 # 20% dari data latih akan digunakan sebagai data validasi untuk early stopping.

# Pembagian data pertama: memisahkan data Latih+Validasi dari data Uji.
# shuffle=False sangat PENTING untuk data deret waktu agar urutan waktu tidak rusak.
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size_ratio, shuffle=False)
# Pembagian data kedua: memisahkan data Latih dari data Validasi.
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size_ratio, shuffle=False)

st.info(f"Ukuran Data: **{len(X_train)}** Latih, **{len(X_val)}** Validasi, **{len(X_test)}** Uji.")

# --- Langkah 4: Normalisasi Fitur ---
# StandardScaler mengubah skala semua fitur agar memiliki rata-rata 0 dan standar deviasi 1.
# Ini membantu model belajar lebih stabil dan cepat.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 'fit_transform' belajar dari data latih dan mengubahnya.
X_val_scaled = scaler.transform(X_val)      # 'transform' hanya mengubah data validasi.
X_test_scaled = scaler.transform(X_test)     # 'transform' hanya mengubah data uji.

# --- Langkah 5 (Opsional): Optimasi Hyperparameter ---
st.sidebar.header("⚙️ Optimasi Hyperparameter")
# Definisikan hyperparameter yang akan diuji.
param_grid = {
    'objective': ['regression_l1'], 'boosting_type': ['gbdt'], 'learning_rate': [0.05, 0.1], 
    'num_leaves': [31, 50], 'max_depth': [5, 10], 'reg_alpha': [0.0, 0.1, 0.5], 
    'reg_lambda': [0.0, 0.1, 0.5], 'min_child_samples': [20, 50]
}
# Checkbox untuk mengaktifkan/menonaktifkan proses GridSearchCV yang lama.
run_gridsearch = st.sidebar.checkbox("Jalankan GridSearchCV", value=False)

# Jika checkbox diaktifkan, jalankan fungsi run_grid_search.
if run_gridsearch:
    best_params = run_grid_search(X_train_scaled, y_train.values.ravel(), param_grid)
    st.subheader("⚙️ Parameter Terbaik Hasil GridSearchCV")
    st.json(best_params)
else:
    # Jika tidak, gunakan parameter terbaik yang sudah ditemukan dari penelitian sebelumnya.
    st.warning("GridSearchCV tidak dijalankan. Menggunakan parameter contoh.")
    best_params = {
        'boosting_type': 'gbdt', 'learning_rate': 0.05, 'max_depth': 5,
        'num_leaves': 31, 'objective': 'regression_l1', 'reg_alpha': 0.1, 
        'reg_lambda': 0.1, 'min_child_samples': 20
    }
    st.subheader("⚙️ Parameter Contoh (GridSearchCV Tidak Dijalankan)")
    st.json(best_params)

# --- Langkah 6: Pelatihan Model Final ---
# Memanggil fungsi untuk melatih model dengan parameter terbaik dan data yang sudah disiapkan.
final_model = train_final_model(best_params, X_train_scaled, y_train.values.ravel(), X_val_scaled, y_val.values.ravel())
st.success("Model final berhasil dilatih!")
# Menyimpan model dan nama kolom ke dalam session_state agar bisa diakses di bagian lain aplikasi.
st.session_state.final_model = final_model
st.session_state.X_columns = X.columns

# --- Langkah 7: Evaluasi Model pada Seluruh Data ---
# Di sini kita melakukan prediksi pada set data Latih+Validasi dan set data Uji
# untuk membandingkan performanya dan menganalisis overfitting.

# Prediksi pada data Latih+Validasi
X_train_val_scaled = scaler.transform(X_train_val) # Normalisasi gabungan data latih+validasi
y_pred_train_val = final_model.predict(X_train_val_scaled)
# Menghitung metrik untuk data Latih+Validasi
rmse_train = np.sqrt(mean_squared_error(y_train_val, y_pred_train_val))
r2_train = r2_score(y_train_val, y_pred_train_val)
mae_train = mean_absolute_error(y_train_val, y_pred_train_val)
mape_train = mean_absolute_percentage_error(y_train_val, y_pred_train_val)

# Prediksi pada data Uji
y_pred_test = final_model.predict(X_test_scaled)
# Menghitung metrik untuk data Uji
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

# --- Langkah 8: Menampilkan Hasil Evaluasi ---
# Menampilkan semua metrik yang dihitung dalam dua kolom agar rapi.
st.subheader("📊 Hasil Evaluasi Model (Sesuai Format Jurnal)")
col_eval1, col_eval2 = st.columns(2)
with col_eval1:
    st.markdown("### 📈 Data Latih (Training Set)")
    st.metric(label="Avg. Error (MAPE)", value=f"{mape_train:.2%}")
    st.write(f"**RMSE:** {rmse_train:.4f}")
    st.write(f"**MAE:** {mae_train:.4f}")
    st.write(f"**R²:** {r2_train:.4f}")
with col_eval2:
    st.markdown("### 📉 Data Uji (Testing Set)")
    st.metric(label="Avg. Error (MAPE)", value=f"{mape_test:.2%}")
    st.write(f"**RMSE:** {rmse_test:.4f}")
    st.write(f"**MAE:** {mae_test:.4f}")
    st.write(f"**R²:** {r2_test:.4f}")

# --- Langkah 9: Visualisasi Hasil Prediksi ---
# Membuat plot grafik untuk membandingkan harga aktual dengan harga prediksi.
st.subheader("📈 Visualisasi Hasil Prediksi pada Data Uji")
fig, ax = plt.subplots(figsize=(12, 6))
test_dates = df.loc[y_test.index, 'Date'] # Mengambil tanggal yang sesuai untuk sumbu-x.
ax.plot(test_dates, y_test.values, label="Harga Aktual", color='blue', marker='.', linestyle='-')
ax.plot(test_dates, y_pred_test, label="Harga Prediksi", color='red', linestyle='--')
ax.legend()
ax.set_title("Perbandingan Harga Aktual vs Prediksi pada Data Uji", fontsize=16)
ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (USD)")
ax.grid(True); fig.autofmt_xdate(); st.pyplot(fig)

# --- Langkah 10: Analisis Kepentingan Fitur ---
# Menampilkan grafik yang menunjukkan fitur mana yang paling berpengaruh
# dalam membuat prediksi menurut model.
st.header("🔬 Analisis Tambahan untuk Jurnal")
st.subheader("🧠 Analisis Kepentingan Fitur (Feature Importance)")
if 'final_model' in st.session_state: # Memastikan model sudah dilatih
    model_to_plot = st.session_state.final_model
    feature_names = st.session_state.X_columns
    feature_importance_df = pd.DataFrame({
        'Fitur': feature_names,
        'Tingkat Kepentingan': model_to_plot.feature_importances_
    }).sort_values(by='Tingkat Kepentingan', ascending=False).head(15) # Mengambil 15 fitur teratas.
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='Tingkat Kepentingan', y='Fitur', ax=ax_imp, palette='viridis')
    ax_imp.set_title('Tingkat Kepentingan Fitur dalam Model LightGBM', fontsize=16)
    st.pyplot(fig_imp)

# ==============================================================================
# 5. FITUR TAMBAHAN: PREDIKSI DATA INDIVIDUAL
# ==============================================================================
# Penjelasan:
# Bagian ini adalah fitur interaktif di sidebar yang memungkinkan pengguna
# memilih satu baris data spesifik dan melihat prediksi model untuk data tersebut.

st.sidebar.header("🔍 Prediksi Data Individual")
if not df.empty:
    # Ambil indeks dari dataframe yang sudah di proses (setelah dropna)
    available_indices = X.index 
    selected_index = st.sidebar.selectbox("Pilih Indeks Data untuk Prediksi", available_indices)
    
    # Ambil baris data yang dipilih oleh pengguna.
    selected_row_processed = df.loc[selected_index]
    
    # Tampilkan informasi dari baris yang dipilih.
    st.sidebar.write(f"**Tanggal:** {selected_row_processed['Date'].strftime('%Y-%m-%d')}")
    st.sidebar.write(f"**Harga Sebenarnya:** {selected_row_processed['Price']:.4f}")
    
    # Ambil hanya fitur-fitur lag dari baris yang dipilih.
    input_features = selected_row_processed[feature_columns]
    st.sidebar.write("**Fitur yang Digunakan:**")
    st.sidebar.json(input_features.to_dict())

    # Jika tombol ditekan, lakukan prediksi.
    if st.sidebar.button("Prediksi Harga untuk Data Terpilih"):
        input_data = np.array([input_features.values]) # Ubah fitur menjadi format yang bisa diterima model.
        input_scaled = scaler.transform(input_data) # Normalisasi data input menggunakan scaler yang sama.
        prediction = final_model.predict(input_scaled) # Lakukan prediksi.
        st.sidebar.success(f"**Prediksi Harga ADA:** ${prediction[0]:.4f}")

