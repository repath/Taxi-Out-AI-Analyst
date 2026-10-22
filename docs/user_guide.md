# User Guide: Taxi-Out Dispatch AI & Operations Dashboard

**Project:** Aviation Taxi-Out Duration Prediction Pipeline  
**Target Audience:** Airline Dispatchers, Ground Controllers, Operations Researchers  
**Submission Identifier:** `2606MGAIL130`  

---

## 1. Quickstart & Deployment Options

### Option A: Launch Interactive Streamlit Application (Recommended)
```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run Streamlit UI
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Run Standalone Python Prediction CLI
```powershell
python predict.py
```

### Option C: Serve Alternative Static HTML Dashboard
```powershell
python -m http.server 8080 --directory dashboard
```
Open `http://localhost:8080`.

---

## 2. Navigating the Streamlit Web Application

The Streamlit dashboard features 6 dedicated operational tabs:

### 2.1 Tab 0: Data Understanding & Quality
- **Data Funnel:** Visualizes the removal of cancellations, diversions, and invalid records.
- **Percentile Summary:** Shows median ($23.4\text{m}$), 90th percentile ($34.6\text{m}$), and 99th percentile ($52.8\text{m}$) benchmarks.
- **Exploratory Patterns:** Breakdown of taxi durations by airline carrier, departure hour waves, and hub airport geometry.

### 2.2 Tab 1: Dispatch Simulator
- **Interactive Flight Configuration:** Adjust carrier, origin airport, departure time, gate delay, and surface congestion.
- **Scenario Presets:**
  - *DFW Morning Peak Push (AA to ORD):* High congestion (18 departures/hr), peak morning bank.
  - *DFW Midday Off-Peak (AA to MIA):* Low congestion (5 departures/hr).
  - *JFK Evening Congested Peak (B6 to LAX):* Constrained geometry, severe queueing.
- **Predicted Outcomes:**
  - Predicted Taxi-Out in minutes.
  - Estimated Wheels-Off Time ($\text{TOT} = \text{DEP\_TIME} + \widehat{\text{TAXI\_OUT}}$).
  - Surface Congestion Risk Badge (Green/Low, Yellow/Moderate, Orange/High, Red/Severe).

### 2.3 Tab 2: Model Performance & Benchmarks
- **Comparison Table:** Side-by-side performance of Linear Regression, Random Forest, and XGBoost.
- **Operational Tolerance:** FAA $\pm 3\text{-minute}$ and $\pm 5\text{-minute}$ compliance percentages.
- **Operational Risk Matrix:** Classification view of predicted vs actual congestion tiers.
- **Feature Importance:** Gini and gain ranking of top surface predictors.

### 2.4 Tab 3: Congestion Analytics
- **Surface Response Curve:** Non-linear taxi delay escalation as concurrent departures increase from 5 to 25+ per hour.
- **Airfield Hub Comparisons:** DFW (large 7-runway layout) vs JFK/LGA (geometry constrained) vs ATL (efficient parallel layout).

### 2.5 Tab 4: AI Analyst & A-CDM Architecture Documentation
- **Interactive AI Analyst (RAG)**: Conversational assistant answering questions grounded directly on project markdown reports with source citations (zero external API cost).
- **A-CDM Knowledge Base**: Complete reference on Airport Collaborative Decision Making, departure metering programs (DMP), and environmental sustainability impact.
- **Searchable Operations Terminology**: Filterable operational definitions and real-world flight examples.

---

## 3. Operational Guidance for Air Traffic Control & Dispatchers

| Surface Congestion | Predicted Taxi-Out | Operational Action / Advisory |
| :--- | :---: | :--- |
| **Low (< 8 dep/hr)** | $< 20\text{ min}$ | Standard pushback clearance. Minimum fuel wasted in surface queue. |
| **Moderate (8–14 dep/hr)** | $20 - 28\text{ min}$ | Monitor departure queues. Standard metering active. |
| **High (15–21 dep/hr)** | $28 - 35\text{ min}$ | Pre-departure gate hold advisory recommended (hold 5–10 min at gate with engines off). |
| **Severe (> 21 dep/hr)** | $> 35\text{ min}$ | Departure Metering Program (DMP) enforced. Aircraft held at gate to prevent ramp gridlock. |
