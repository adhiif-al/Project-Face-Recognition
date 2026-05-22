# 🎭 Face Recognition System => Nilai Eigen & Eigenface

> **Aplikasi Pengenalan Wajah berbasis PCA/Eigenface**  
> Implementasi metode *Principal Component Analysis* (PCA) menggunakan Singular Value Decomposition (SVD) untuk pengenalan wajah secara real-time maupun berbasis gambar.

---

## 📌 Deskripsi Proyek

Proyek ini merupakan implementasi pengenalan wajah (*face recognition*) menggunakan konsep **Nilai Eigen** dan **Eigenface** yang merupakan aplikasi nyata dari **Aljabar Linear**. Sistem ini dibangun dengan pendekatan matematis murni menggunakan **SVD (Singular Value Decomposition)** tanpa bergantung pada *library* machine learning eksternal.

Program memungkinkan pengguna untuk:
- Melatih model dari dataset foto wajah
- Mengenali wajah dari **file gambar** maupun **webcam langsung**
- Memvisualisasikan komponen-komponen eigen, mean face, dan evaluasi akurasi model

---

## 🧮 Landasan Matematika

Metode eigenface bekerja dengan prinsip matematis berikut:

| Konsep | Penjelasan |
|--------|-----------|
| **Mean Face** | Rata-rata semua gambar wajah dalam dataset |
| **Centering** | Pengurangan setiap gambar dengan mean face |
| **SVD** | Dekomposisi matriks untuk mencari eigenvector dominan |
| **Eigenface** | Basis vektor yang merepresentasikan pola variasi wajah |
| **Proyeksi** | Representasi wajah pada ruang eigenface |
| **Jarak Euclidean** | Ukuran kemiripan antar proyeksi wajah |

### Persamaan Inti

```
A · x = λ · x
```

Dimana:
- **A** = matriks kovarians wajah
- **x** = eigenvector (eigenface)
- **λ** = eigenvalue (nilai eigen)

---

## 🖥️ Tampilan Aplikasi

### Main Interface
```

```

---

## 📁 Struktur Proyek

```
Face-Recognition-Eigenface/
│
├── backend.py          # Core PCA/SVD engine & recognition logic
├── frontend.py         # GUI aplikasi (CustomTkinter)
├── visualisasi.py      # Panel visualisasi eigen & evaluasi akurasi
│
├── dataset/            # Folder dataset (buat sendiri)
│   ├── nama_orang_1/
│   │   ├── foto1.jpg
│   │   └── foto2.jpg
│   ├── nama_orang_2/
│   │   └── foto1.jpg
│   └── ...
│
└── README.md
```

---

## ⚙️ Cara Penggunaan

### 1. Siapkan Dataset

Buat struktur folder dataset seperti berikut. **Nama subfolder = label identitas orang**.

```
dataset/
├── Budi/
│   ├── budi_1.jpg
│   └── budi_2.jpg
├── Siti/
│   ├── siti_1.jpg
│   └── siti_2.jpg
```

> Format yang didukung: `.jpg`, `.jpeg`, `.png`

### 2. Jalankan Aplikasi

```bash
python frontend.py
```

### 3. Alur Penggunaan

```
1. Klik [Choose Folder]   →  Pilih folder dataset
   └─ Model otomatis dilatih via SVD

2. Pilih sumber uji:
   ├─ Browse files         →  Upload gambar dari komputer
   └─ Use Webcam ☑        →  Buka jendela kamera

3. Atur threshold          →  Geser slider (default: 5000)

4. Klik [Start Face Recognition]
   └─ Hasil & akurasi tampil di workspace

5. Klik [📊 Lihat Visualisasi]
   └─ Buka panel analisis eigen
```

---

## 📊 Panel Visualisasi

Akses melalui tombol **"📊 Lihat Visualisasi"** setelah dataset dimuat.

| Tab | Isi |
|-----|-----|
| 🖼 Eigenface | Mean face + grid 9 eigenface teratas |
| 📊 Grafik Nilai Eigen | Plot λ tiap komponen PCA vs threshold |
| 🧭 Visualisasi Vektor Eigen | Ilustrasi grafis `A·x = λ·x` (4 kasus skalar) |
| 📈 Plot Eigenface Pertama | Distribusi nilai piksel vektor eigen ke-1 |
| 🎯 Evaluasi Akurasi | LOOCV accuracy + Confusion Matrix |

---

## 🔧 Instalasi

### Prasyarat

- Python 3.8+
- pip

### Install Dependensi

```bash
pip install opencv-python numpy Pillow customtkinter matplotlib
```

Atau menggunakan file requirements:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
opencv-python>=4.5
numpy>=1.21
Pillow>=9.0
customtkinter>=5.0
matplotlib>=3.5
```

---

## 🛠️ Detail Teknis

### `backend.py` — FaceRecognitionBackend

| Metode | Fungsi |
|--------|--------|
| `train()` | Memuat dataset, menghitung mean face, menjalankan SVD |
| `recognize(path)` | Pengenalan dari file gambar |
| `recognize_from_frame(frame)` | Pengenalan dari frame BGR (webcam) |
| `_project_and_match()` | Proyeksi ke ruang eigenface + pencocokan jarak Euclidean |

**Alur Training:**
```
Gambar → Grayscale → Resize (100×100) → Flatten
       → Hitung Mean Face
       → Centering (X - μ)
       → SVD: U, S, Vt = svd(centered.T)
       → Eigenfaces = U
       → Bobot = EigenFaces.T × Centered.T
```

**Alur Recognition:**
```
Test Image → Grayscale → Resize → Flatten
           → Centering
           → Proyeksi: w_test = Eigenfaces.T × (x - μ)
           → Euclidean Distance ke semua bobot training
           → Best match (jika jarak < threshold)
```

### `frontend.py` — FaceApp (CustomTkinter GUI)

- Dark theme dengan skema warna `#09090b` / `#18181c`
- Thread terpisah untuk feed webcam (≈30 fps) agar UI tidak freeze
- Threshold dapat disesuaikan realtime via slider (1000–10000)
- Menampilkan akurasi dan waktu komputasi di status bar

### `visualisasi.py` — VisualisasiWindow

- 5 tab interaktif berbasis Matplotlib + TkAgg backend
- LOOCV (Leave-One-Out Cross Validation) dijalankan langsung di GUI
- Confusion matrix dengan heatmap manual (`imshow`)
- Threshold dapat diubah dari dalam jendela visualisasi tanpa restart

---

## 📐 Penjelasan Threshold

Threshold adalah batas **jarak Euclidean** maksimum antara proyeksi wajah uji dengan wajah di database.

```
Jarak < Threshold  →  Wajah dikenali (match)
Jarak ≥ Threshold  →  "Unknown" (tidak dikenali)
```

> **Tips:** Mulai dari threshold `5000`. Naikkan jika terlalu banyak hasil "Unknown", turunkan jika terlalu banyak hasil salah.

---

## 🧠 Konsep Eigenvalue & Eigenvector

```
Eigenvalue besar  →  Komponen yang menjelaskan variasi terbesar
Eigenvalue kecil  →  Komponen yang kurang penting (noise)

Eigenface #1  →  Pola pencahayaan global
Eigenface #2  →  Variasi arah hadap
Eigenface #n  →  Detail semakin halus
```

Dalam visualisasi **Grafik Nilai Eigen**, komponen di atas garis threshold (λ²) dianggap dominan dalam proses pengenalan.

---

## 👨‍💻 Teknologi yang Digunakan

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-SVD-orange?style=flat-square&logo=numpy)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-GUI-purple?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisasi-red?style=flat-square)

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan **Project Based Learning — Aljabar Linear**.  
Bebas digunakan dan dimodifikasi untuk tujuan akademik dengan menyertakan atribusi.

---

<p align="center">
  <i>Face Recognition System © 2025 — Powered by PCA & Eigenface</i>
</p>
