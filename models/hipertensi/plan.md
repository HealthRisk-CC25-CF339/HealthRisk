## 1. Definisi Masalah & Tujuan

- **Problem Statement**
    
    - Prediksi status hipertensi (0 = tidak, 1 = ya) pasien berdasarkan 13 fitur klinis dari dataset BRFSS 2015.
        
- **Objective**
    
    - Membangun model klasifikasi diskriminatif “from scratch” dengan TensorFlow (Keras API) yang mencapai metrik AUC–ROC ≥ 0.80, dan siap diekspor untuk inference.
        
- **Success Criteria**
    
    - AUC–ROC, precision, recall, F1 pada test set; model ringan (< 10 MB) dan inference latency < 50 ms per instance.
        

---

## 2. Data Understanding & Exploratory Data Analysis

1. **Inventarisasi Fitur**
    
    - Numerik: usia, tekanan darah, kolesterol, max HR, ST-depression, jumlah pembuluh.
        
    - Kategorikal: jenis nyeri dada, FBS>120, ECG result, exercise angina, slope, defect type.
        
    - Target: hipertensi (0/1), kelas seimbang.
        
2. **Statistik Deskriptif**
    
    - Distribusi, mean/median, skewness, range tiap fitur.
        
    - Korelasi (heatmap) untuk deteksi multi-collinearity.
        
3. **Deteksi Masalah Data**
    
    - Missing values & outlier (boxplot, IQR).
        
    - Imputasi (median untuk numerik; mode/“unknown” untuk kategori).
        
4. **Visualisasi Insight**
    
    - Histogram, violin plot per kelas target.
        
    - Pairwise scatter untuk fitur kunci (misal: BP vs. kolesterol).
        

---

## 3. Preprocessing & Feature Engineering

1. **Scaling & Normalisasi**
    
    - StandardScaler (z-score) untuk fitur numerik agar converge lebih cepat.
        
2. **Encoding Kategori**
    
    - One-hot encoding atau embedding sederhana di model untuk fitur multi-kelas (chest pain, ECG, slope, defect).
        
3. **Feature Synthesis (opsional)**
    
    - Interaksi BP×usia atau kolesterol×FBS untuk capture nonlinearitas.
        
4. **Data Split**
    
    - Stratified split:
        
        - Training 70%
            
        - Validation 15%
            
        - Test 15%
            

---

## 4. Baseline & Benchmark

- **Model Simple**
    
    - Logistic Regression atau Decision Tree (scikit-learn) sebagai tolok ukur performa.
        
- **Metrik Benchmark**
    
    - Catat AUC–ROC, akurasi, precision/recall pada validation set.
        

---

## 5. Desain & Training Model TensorFlow

1. **Arsitektur MLP**
    
    - Input layer sesuai dimensi fitur.
        
    - 2–4 Hidden Dense layers (32–128 neuron), aktivasi ReLU.
        
    - Regularisasi: Dropout (0.2–0.5), L2 weight decay.
        
    - Output single neuron + sigmoid.
        
2. **Kompilasi**
    
    - Loss: `binary_crossentropy`
        
    - Optimizer: Adam (lr tunable)
        
    - Metrik: AUC, accuracy, precision, recall
        
3. **Callbacks**
    
    - EarlyStopping (monitor val_auc, patience 5)
        
    - ModelCheckpoint (best val_auc)
        
    - (Opsional) CSVLogger / TensorBoard untuk monitoring
        
4. **Hyperparameter Tuning**
    
    - Grid/Random Search pada:
        
        - Jumlah layer & neuron
            
        - Learning rate (1e-4–1e-2)
            
        - Dropout rate
            
        - Batch size (16, 32, 64)
            
    - Gunakan validation set atau K-fold CV (stratified k=5)
        

---

## 6. Evaluasi Akhir & Interpretabilitas

1. **Performance on Test Set**
    
    - Report: AUC–ROC curve, confusion matrix, classification report (precision, recall, F1).
        
2. **Calibration & Robustness**
    
    - Reliability diagram untuk kalibrasi probabilitas.
        
    - Uji sensitivitas dengan noise injection kecil (±5% fitur numerik).
        
3. **Interpretasi Model**
    
    - SHAP values atau LIME untuk insight fitur paling berdampak.
        
    - Partial Dependence Plot untuk relasi fitur–target.
        

---

## 7. Model Export & Inference Packaging

1. **Export Format**
    
    - `tf.saved_model` untuk keperluan Python/Node.js
        
    - (Opsional) Konversi ke TFLite atau TF.js format jika stack target butuh
        
2. **Interface Inference**
    
    - JSON schema: array 13 fitur ter-standarisasi
        
    - Modul loading model + preprocessing pipeline yang sama dengan training
        

---

## 8. Deployment Readiness & Monitoring

1. **Deployment Handover**
    
    - Sediakan artefak: model file, scaler parameters, README integration.
        
2. **Monitoring & Maintenance**
    
    - Rencana retraining: trigger saat data real drift terdeteksi (misal setelah 6 bulan).
        
    - Logging inference: request features + prediksi untuk audit & drift analysis.
        

---

**9. Deliverables & Timeline (3 Minggu)**

| Periode                               | Aktivitas                                                                                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Minggu 1 (May 27 – June 2, 2025)**   | – Data EDA & laporan deskriptif<br>– Implementasi preprocessing pipeline & split data<br>– Baseline model (Logistic Regression)           |
| **Minggu 2 (June 3 – June 9, 2025)**   | – Pengembangan MLP awal di TensorFlow<br>– Setup callbacks (EarlyStopping, Checkpoint)<br>– Eksperimen awal & evaluasi di validation set  |
| **Minggu 3 (June 10 – June 13, 2025)** | – Hyperparameter tuning & final evaluation di test set<br>– Interpretability analysis (SHAP/LIME)<br>– Model export & packaging artefak inference |
