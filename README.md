# 📊 Aplikasi Web Analisis Klasterisasi Produksi Perikanan (K-Medoids)

Aplikasi ini melakukan analisis dan klasterisasi terhadap dataset hasil _Produksi Perikanan Tangkap di Provinsi Jawa Barat_ menggunakan algoritma _K-Medoids_ dan visualisasi berbasis web dengan **Flask**.

## 🐟 Fitur

- Membaca dataset Excel produksi perikanan.
- Melakukan klasterisasi menggunakan K-Medoids dengan 3 klaster (Rendah, Sedang, Tinggi).
- Menampilkan data asli dan hasil klaster dalam tabel HTML.
- Visualisasi 2D hasil PCA dari data produksi.
- Menyediakan tombol untuk mengunduh hasil clustering dalam format Excel.

---

## 🛠️ Cara Instalasi

1. Clone repositori (atau simpan file `main.py` dan template-nya)

```bash
git clone https://github.com/AdityaWjy/Flask-Kmedoids-Project
cd Flask-Kmedoids-Project

# Buat dan aktifkan virtual environment
python -m venv venv
source venv/bin/activate        # untuk Linux/macOS
venv\Scripts\activate           # untuk Windows

# Install dependensi
pip install flask pandas numpy scikit-learn matplotlib openpyxl

# Jalankan project
cd SPK_FLASK
cd src
python main.py
http://localhost:5000



```
