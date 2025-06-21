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

# Masuk ke direktori project:

1. cd SPK_FLASK

# python -m venv venv

2. python -m venv venv

# Aktifkan virtual environment:

# Windows:

3. venv\Scripts\activate

# Macos

3. source venv/bin/activate

# Install Dependensi

4. pip install flask pandas numpy scikit-learn matplotlib openpyxl

# Jalankan aplikasi

5. cd src
6. python main.py
7. http://127.0.0.1:5000/ # tergantung diterminal munculnya apa

8. Clone repositori (atau simpan file `main.py` dan template-nya)
