# Laporan Mingguan Proyek Hipertensi - Minggu 1

**Periode:** 27 Mei – 2 Juni 2025

## 1. Tujuan Minggu Ini
Sesuai dengan `plan.md`, tujuan untuk Minggu 1 adalah:
- Melakukan Exploratory Data Analysis (EDA) dan membuat laporan deskriptif.
- Mengimplementasikan pipeline preprocessing dan melakukan pemisahan data (split data).
- Membuat model baseline menggunakan Logistic Regression.

## 2. Ringkasan Exploratory Data Analysis (EDA)
Analisis dilakukan pada dataset `hypertension_data.csv`. Berikut adalah temuan utama:

-   **Struktur Data**: Dataset terdiri dari 26.083 entri dan 14 kolom, termasuk 13 fitur dan 1 kolom target bernama `target`.

### Deskripsi Fitur Detail
Berikut adalah deskripsi untuk setiap fitur dalam dataset `hypertension_data.csv`:
-   `age`: Usia pasien (dalam tahun).
-   `sex`: Jenis kelamin pasien (1: male; 0: female).
-   `cp` (Chest Pain Type): Jenis nyeri dada.
    -   0: Asymptomatic
    -   1: Typical angina
    -   2: Atypical angina
    -   3: Non-anginal pain
-   `trestbps` (Resting Blood Pressure): Tekanan darah istirahat (dalam mm Hg).
-   `chol` (Serum Cholesterol): Kolesterol serum (dalam mg/dl).
-   `fbs` (Fasting Blood Sugar > 120 mg/dl): Apakah gula darah puasa pasien > 120 mg/dl (1: ya; 0: no).
-   `restecg` (Resting ECG Results): Hasil ECG istirahat.
    -   0: Normal
    -   1: ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    -   2: Probable or definite left ventricular hypertrophy by Estes' criteria
-   `thalach` (Maximum Heart Rate Achieved): Denyut jantung maksimum yang dicapai.
-   `exang` (Exercise Induced Angina): Angina yang diinduksi oleh olahraga (1: ya; 0: no).
-   `oldpeak` (ST Depression): Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat.
-   `slope` (Slope of the Peak Exercise ST Segment): Kemiringan segmen ST puncak saat olahraga.
    -   0: Upsloping
    -   1: Flat
    -   2: Downsloping
-   `ca` (Number of Major Vessels): Jumlah pembuluh darah utama (0–3) yang diwarnai oleh flourosopy.
-   `thal` (Thalassemia): Jenis kelainan darah Thalassemia.
    -   Value 3: Normal
    -   Value 6: Fixed defect
    -   Value 7: Reversable defect
    *(Catatan: Dataset CSV awalnya memiliki nilai 0, 1, 2, 3 untuk 'thal'. Nilai '0' (0.59% kasus) diimputasi dengan modus yang dihitung secara dinamis dari nilai non-0 lainnya (yaitu nilai '2') selama preprocessing. Setelah imputasi, nilai yang ada adalah 1, 2, 3.)*
-   `target`: Apakah pasien memiliki hipertensi (1: ya; 0: no).

### Ringkasan Temuan EDA Lanjutan
-   **Missing Values**:
    -   Kolom `sex` memiliki 25 nilai yang hilang. Ini ditangani dengan imputasi modus oleh `SimpleImputer` dalam pipeline.
    -   Fitur lainnya tidak memiliki missing values.
-   **Tipe Data Fitur**:
    -   **Numerik**: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`.
    -   **Kategorikal** (meskipun beberapa direpresentasikan sebagai angka): `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`.
-   **Distribusi Fitur Kunci**:
    -   `age`: Rentang usia pasien cukup luas (11-98 tahun), dengan rata-rata sekitar 55.6 tahun.
    -   `thal`: Sebelum imputasi, nilai '0' terdapat pada 0.59% data. Setelah imputasi nilai '0' dengan modus dinamis ('2'), distribusi menjadi (berdasarkan output notebook):
        -   `2`: 55.641606%
        -   `3`: 38.707204%
        -   `1`: 5.651190%
    -   `target` (Hipertensi): Distribusi kelas target cukup seimbang:
        -   Kelas 1 (Hipertensi): sekitar 54.7%
        -   Kelas 0 (Tidak Hipertensi): sekitar 45.3%
-   **Korelasi Antar Fitur Numerik dan Target** (berdasarkan output heatmap):
    -   `thalach` (Max Heart Rate) memiliki korelasi positif sedang dengan `target` (0.42).
    -   `oldpeak` (ST Depression) memiliki korelasi negatif sedang dengan `target` (-0.44).
    -   `trestbps` (Resting Blood Pressure) memiliki korelasi negatif lemah dengan `target` (-0.15).
    -   Fitur numerik lainnya (`age`, `chol`) menunjukkan korelasi yang sangat lemah dengan `target`.
    -   Antar fitur, `thalach` dan `oldpeak` menunjukkan korelasi negatif sedang (-0.34).
-   **Catatan Khusus pada Fitur Kategorikal (selain `thal` yang sudah dibahas)**:
    -   Fitur `ca` (Number of Major Vessels) dalam data CSV memiliki nilai hingga 4. Deskripsi awal menyebutkan rentang 0-3. Nilai 4 akan diperlakukan sebagai kategori terpisah oleh OneHotEncoder.

## 3. Preprocessing dan Pemisahan Data
-   **Imputasi Khusus**:
    -   Nilai '0' pada fitur `thal` (0.59% kasus) diimputasi dengan modus yang dihitung secara dinamis dari nilai non-0 lainnya (yaitu nilai '2') sebelum pemisahan data dan pipeline preprocessing utama.
-   **Preprocessing Pipeline**:
    -   Untuk fitur numerik: Imputasi median (`SimpleImputer`) diikuti oleh `StandardScaler`.
    -   Untuk fitur kategorikal: Imputasi modus (`SimpleImputer`) diikuti oleh `OneHotEncoder` (handle_unknown='ignore').
-   **Pemisahan Data**: Data dibagi menjadi tiga set secara stratified berdasarkan kolom `target`:
    -   Training set: 70% (18.258 sampel)
    -   Validation set: 15% (3.912 sampel)
    -   Test set: 15% (3.913 sampel)
-   **Hasil Preprocessing**: Setelah preprocessing (khususnya One-Hot Encoding), jumlah fitur menjadi 30.

## 4. Model Baseline (Logistic Regression)
Model Logistic Regression dilatih pada data training yang telah diproses dan dievaluasi pada validation set.

-   **Performa pada Validation Set**:
    -   **AUC-ROC**: 0.9270
    -   **Accuracy**: 0.8704
    -   **Precision (kelas 1)**: 0.8488
    -   **Recall (kelas 1)**: 0.9285
    -   **F1-score (kelas 1)**: 0.8869
    -   **Classification Report Keseluruhan**:
        ```
                      precision    recall  f1-score   support

                   0       0.90      0.80      0.85      1771
                   1       0.85      0.93      0.89      2141

            accuracy                           0.87      3912
           macro avg       0.88      0.86      0.87      3912
        weighted avg       0.87      0.87      0.87      3912
        ```

## 5. Artefak yang Dihasilkan untuk Minggu Berikutnya
Untuk kelanjutan pekerjaan di Minggu 2, artefak berikut telah disimpan di direktori `models/hipertensi/processed_data/`:
-   `preprocessor.joblib`: Objek preprocessor yang telah di-fit.
-   `X_train_processed.csv`: Data fitur training yang sudah diproses.
-   `X_val_processed.csv`: Data fitur validasi yang sudah diproses.
-   `X_test_processed.csv`: Data fitur test yang sudah diproses.
-   `y_train.csv`: Label target untuk data training.
-   `y_val.csv`: Label target untuk data validasi.
-   `y_test.csv`: Label target untuk data test.
