GitHub README
>>>>>>> origin/mainPS D:\JNTU CourseWork\Taxi-In-Out_Prediction> git add README.md
>> git commit -m "Resolve README merge conflict"
>> git push -u origin main
[main a502079] Resolve README merge conflict
Enumerating objects: 65, done.
Counting objects: 100% (65/65), done.
Delta compression using up to 12 threads
Compressing objects: 100% (60/60), done.
Writing objects: 100% (63/63), 15.03 MiB | 675.00 KiB/s, done.
Total 63 (delta 2), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (2/2), done.
To https://github.com/repath/Predicting-Aircraft-Taxi-Out-Duration-.git
   32dd674..a502079  main -> main
branch 'main' set up to track 'origin/main'.
# Aviation Taxi-Out Time Prediction Pipeline
**Coursework Project: JNTU Aviation Data Science & Operations Research**  
**Dataset:** BTS Marketing Carrier On-Time Performance (2025 Full Year)  
**Primary Hub Scope:** Dallas/Fort Worth International Airport (DFW) & National Airspace System  

---

## Overview
This repository provides a complete, production-ready machine learning framework for predicting **Taxi-Out Duration** (`TAXI_OUT = WHEELS_OFF - DEP_TIME`) in minutes, following standard Airport Collaborative Decision Making (A-CDM) operating principles.

## Architecture Change Backup

Before proceeding with architecture changes, a complete project backup was created on **2026-10-20**.

- **Backup name:** `before arch change`
- **Location:** `C:\tmp\before arch change`
- **Contents:** Complete project directory, including `.git`, `.venv`, data, models, outputs, and current working changes.
- **Verification:** 32,805 files copied with no copy failures.

## Project Structure
- `data/`
  - `raw/`: Raw BTS on-time performance CSV records.
  - `processed/`: Cleaned dataset, time-split training and testing feature sets, and cleaning audit logs.
- `figures/`: High-resolution evaluation charts (distribution, congestion response, model comparison, parity scatter, feature importances).
- `models/`: Serialized models (`linear_regression_baseline.joblib`, `random_forest_regressor.joblib`), target encoder tables, and metric summaries.
- `notebooks/`: Fully executed Jupyter Notebook `Taxi_Out_Prediction_Pipeline.ipynb`.
- `dashboard/`: Modern interactive web dashboard with real-time flight dispatch simulator, Chart.js analytics, and test feed.
- `src/`: Modular Python pipelines for data generation, preprocessing, feature engineering, model training, and evaluation.
- `predict.py`: Inference script for single-flight and batch predictions.
- `Project_Report.md`: Full academic & operational report.

## Quickstart

### 1. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install requirements inside .venv
pip install -r requirements.txt
```

### 2. Launch Streamlit User Interface (Recommended)
```bash
# Run Streamlit UI
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501` to access the interactive Flight Dispatch Simulator, Model Benchmarks, and Airfield Congestion Analytics.

### 3. Run Pipeline Scripts & Inference
```bash
# Run single-flight & scenario predictions
python predict.py

# Re-run ML data preprocessing & feature engineering pipeline
python src/data_preprocessing.py
python src/feature_engineering.py

# Retrain models & generate plots
python src/train_models.py
python src/evaluate.py
```

### 4. Alternative HTML Dashboard Server
```bash
python -m http.server 8080 --directory dashboard
```
Open [http://localhost:8080](http://localhost:8080) for the static HTML/Chart.js visual dashboard.


## Model Results Summary (Nov–Dec Out-of-Time Test Set)

| Metric | Linear Regression Baseline | Random Forest Regressor | XGBoost Regressor (Best) |
| :--- | :---: | :---: | :---: |
| **MAE** | 4.17 minutes | 4.04 minutes | **3.88 minutes** |
| **RMSE** | 10.04 minutes | 10.00 minutes | **9.89 minutes** |
| **$R^2$ Score** | 0.0166 | 0.0239 | **0.0452** |
| **Within $\pm 3$ min** | 54.6% | 57.5% | **58.8%** |
| **Within $\pm 5$ min** | 78.9% | 82.6% | **84.9%** |

*Note: Models evaluated on held-out out-of-time test set (Months 11–12, 12,001 flights) following A-CDM operational thresholds.*
=======
# Predicting-Aircraft-Taxi-Out-Duration-
JNTU Project on AIML
>>>>>>> origin/main
