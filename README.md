# ☀️ Aditya-L1 Solar Flare AI Nowcasting & Forecasting Engine

An end-to-end multi-payload space weather algorithmic pipeline built for ISRO's **Aditya-L1 mission** data from the **ISSDC PRADAN** portal.

The system fuses time-series data from two primary X-ray instruments:
1. **SoLEXS (Solar Low Energy X-ray Spectrometer):** Soft X-rays ($1\text{--}30\text{ keV}$), sensitive to thermal plasma heating and pre-flare coronal conditions.
2. **HEL1OS (High Energy L1 Orbiting X-ray Spectrometer):** Hard X-rays ($10\text{--}150\text{ keV}$), sensitive to non-thermal electron acceleration during impulsive flare phases.

---

## 🚀 Key Features

* **Direct PRADAN 3-Zip File Ingestion (Zero Manual Extraction):**
  * Accepts `1x SoLEXS 24-hour daily zip` + `2x HEL1OS 12-hour zips` directly in-memory.
  * Automatically detects FITS HDU structures, timing headers, and energy channels across CZT and CDTE detectors.
* **Smart Time Stitching & Resampling:**
  * Concatenates morning ($00\text{--}12\text{ UTC}$) and afternoon ($12\text{--}24\text{ UTC}$) HEL1OS streams.
  * Resamples SoLEXS and HEL1OS onto a synchronized UTC time-grid ($10\text{s}$, $30\text{s}$, or $60\text{s}$ cadence).
* **Cross-Payload Physics Engine:**
  * Dynamic quiescent background/baseline extraction.
  * **Spectral Hardness Ratio ($HR = \frac{\text{HEL1OS}}{\text{SoLEXS}}$):** Diagnostic indicator of non-thermal acceleration.
  * **Neupert Effect Gradient ($\frac{dF_{\text{SoLEXS}}}{dt}$):** Rate of thermal emission rise correlated with hard X-ray bursts.
* **Dual-Head AI Engine:**
  * **Nowcasting & Event Segmentation:** Automatic detection of flare boundaries (start, peak, end), duration, fluence, and classification ($B, C, M, X$).
  * **Predictive Forecasting:** HistGradientBoosting models predicting flare probability $1\text{ to }2\text{ hours ahead}$ (**0.971 ROC-AUC** on 20-day dataset).
* **Interactive Mission Dashboard:**
  * Dark-themed astronomy UI with multi-panel Plotly synchronized charts.
  * Space Weather Advisory & Alert cards (Radio Blackout, GPS Scintillation, Satellite Operations).
  * 1-Click Export of synchronized 24h CSV matrices and JSON Event Catalogs.

---

## 📁 Project Structure

```
solar_flare_ai/
├── src/
│   ├── data_loader.py       # PRADAN Zip parsing, FITS extraction & timeline sync
│   ├── physics_engine.py    # Baseline subtraction, Hardness Ratio, Neupert effect, Flare detection
│   └── ai_model.py          # AI machine learning forecasting models & serialization
├── models/
│   ├── solar_flare_ai_model.pkl   # Pre-trained AI model (20 days dataset)
│   └── training_summary.json     # Training metrics (0.971 ROC-AUC) & event catalog
├── app.py                   # Streamlit web application & Plotly dashboard
├── train_pipeline.py        # Automated 20-day training & validation pipeline
└── README.md                # Documentation & usage guide
```

---

## 🏃 How to Run

### 1. Launch the Web Application
Run the following command in your terminal:

```bash
cd C:\Users\sarth\.gemini\antigravity\scratch\solar_flare_ai
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 2. Using the Dashboard
- **Mode 1 (Upload PRADAN Zips):** Drag and drop 1 SoLEXS zip and 2 HEL1OS zips directly into the upload zones in the sidebar.
- **Mode 2 (Explore 20-Day Mission Archive):** Use the date selector to instantly inspect any historical day between `2026-07-18` and `2026-08-06`.

### 3. Retraining the AI Pipeline
If you add more observation days to your data folder, retrain the models by running:

```bash
python train_pipeline.py
```

---

## 📊 Dataset & Model Benchmark
* **Training Corpus:** 20 Days of continuous Aditya-L1 observations ($172,799$ synchronized 10-second timesteps).
* **Detected Events:** $224$ solar flares categorized ($B, C, M, X$).
* **Model Performance:** **0.971 ROC-AUC** on 1-hour ahead predictive horizon.
