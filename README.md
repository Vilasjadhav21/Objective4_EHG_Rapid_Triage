# Objective 4: Rapid Triage (1-Minute Algorithm)

A Python research project for early Term/Preterm prediction and rapid triage using 1-minute uterine **Electrohysterogram (EHG)** recordings.

---

## 🎯 Target Pipeline Overview

```
1-minute EHG recording 
   └──> Minimal Preprocessing 
          └──> EMD (Empirical Mode Decomposition) 
                 └──> IMF-1 Extraction 
                        └──> Top 5 EHG Features 
                               └──> Fine Gaussian Kernel SVM 
                                      └──> Term / Preterm Prediction
```

> **Note**: In this initial setup phase, project structure and environment files have been created. Preprocessing, EMD decomposition, feature extraction, and SVM classification models will be implemented in subsequent phases.

---

## 📊 Dataset Specifications: PhysioNet TPEHGDB v1.0.1

The project utilizes the **PhysioNet Term-Preterm Electrohysterogram Database (TPEHGDB v1.0.1)**. Key specifications:

- **Total Records**: 300 EHG records obtained from 300 unique pregnancies.
- **EHG Channels**: 3 EHG channels (`S1`, `S2`, `S3`).
- **Sampling Frequency**: $20\text{ Hz}$.
- **Record Duration**: 30 minutes per record ($1,800\text{ seconds}$).
- **Signal Channels**: Each record contains 12 signal channels:
  1. `S1 raw`
  2. `S1 0.08–4 Hz`
  3. `S1 0.3–3 Hz`
  4. `S1 0.3–4 Hz`
  5. `S2 raw`
  6. `S2 0.08–4 Hz`
  7. `S2 0.3–3 Hz`
  8. `S2 0.3–4 Hz`
  9. `S3 raw`
  10. `S3 0.08–4 Hz`
  11. `S3 0.3–3 Hz`
  12. `S3 0.3–4 Hz`
- **Filtering Boundary Rule**: For filtered signals, the first and last **180 seconds** must be ignored to avoid transient filter boundary artifacts.
- **Clinical Metadata**: Term/Preterm target metadata is stored in `tpehgdb.smr`.

---

## 📁 Research Project Structure

```
Objective4_EHG_Rapid_Triage/
│
├── data/                    # PhysioNet TPEHGDB raw dataset & metadata (tpehgdb.smr)
│
├── src/                     # Core Python research modules
│   ├── __init__.py          # Package initializer
│   ├── config.py            # Central dataset and pipeline constants
│   ├── data_loader.py       # TPEHGDB signal and header reader stubs
│   ├── metadata.py          # Clinical metadata parser for tpehgdb.smr
│   ├── preprocessing.py     # Minimal preprocessing module stub
│   ├── emd_processing.py    # EMD & IMF-1 decomposition stub
│   ├── feature_extraction.py# Top 5 EHG feature extraction stub
│   └── model.py             # Fine Gaussian Kernel SVM model stub
│
├── notebooks/               # EDA and experimental Jupyter notebooks
│
├── outputs/                 # Pipeline execution outputs
│   ├── plots/               # Generated signal, IMF, & ROC plots
│   ├── imfs/                # Saved Intrinsic Mode Functions data
│   ├── features/            # Extracted feature tables (.csv / .parquet)
│   └── models/              # Saved model checkpoints (.joblib)
│
├── tests/                   # Test suite
│
├── requirements.txt         # Project dependencies
└── README.md                # Technical documentation
```

---

## ⚙️ Setup & Environment

1. **Navigate to directory**:
   ```bash
   cd Objective4_EHG_Rapid_Triage
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⏳ Project Status

- [x] Project structure created
- [x] Environment and configuration set up (`requirements.txt`, `README.md`, `src/config.py`)
- [ ] Minimal Preprocessing implementation
- [ ] Empirical Mode Decomposition (EMD & IMF-1) implementation
- [ ] Top 5 EHG feature extraction implementation
- [ ] Fine Gaussian Kernel SVM classifier implementation & evaluation
