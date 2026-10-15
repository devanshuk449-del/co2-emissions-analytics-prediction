# 🌍 IEA–EDGAR CO₂ Emissions Intelligence Platform

## 📖 Overview
The **IEA–EDGAR CO₂ Emissions Intelligence Platform** is a data-driven analytics and visualization project built with **Python, Pandas, Scikit-learn, Plotly, and Streamlit**.  
It leverages the **IEA–EDGAR dataset (1970–2023)** to provide insights into global, country-level, and sector-level CO₂ emissions, along with machine learning–based forecasting and scenario simulations.

---

## 📂 Project Structure
IEA-EDGAR-CO2-Intelligence/
│
├── data/                # Processed datasets (annual, monthly, scenarios)
├── models/              # Saved ML models (ridge, random forest, etc.)
├── outputs/             # Analytical exports (comparisons, projections)
├──.ipynb checkpoints    # Jupyter notebooks for preprocessing & analysis
├── app.py               # Streamlit dashboard application and all models 
└── README.md            # Project documentation

---

## ⚙️ Features
- **Data Preprocessing**
  - Cleans and transforms raw monthly CO₂ emissions data (1970–2023).
  - Wide → long format conversion, time-series features, anomaly detection.
- **Analytics**
  - Global annual emissions trend visualization.
  - Country-level and sector-level breakdowns.
  - Year-over-year growth analysis (largest increases/decreases).
- **Machine Learning**
  - Ridge Regression, Random Forest, Gradient Boosting models.
  - Model comparison and performance metrics (MAE, RMSE, R²).
- **Interactive Dashboard (Streamlit)**
  - Executive overview with KPIs and historical trends.
  - Global trends, country intelligence, sector intelligence.
  - ML forecasting and scenario simulator modules.
  - Data explorer for transparency.

---

## 🚀 Getting Started

### 1. Clone the Repository

bash
git clone https://github.com/devanshuk449-del/co2-emissions-analytics-prediction.git
cd co2-emissions-analytics-prediction

2. Install Dependencies
bash
pip install -r requirements.txt

4. Run Preprocessing Notebook
Ensure the raw dataset (IEA_EDGAR_CO2_m_1970_2023.csv) is placed in the project directory.
Run the Jupyter notebook to generate processed datasets:

bash
jupyter notebook

4. Launch Streamlit Dashboard
bash
streamlit run app.py

📊 Example Outputs
Global CO₂ Emissions Trend (1970–2023)

Top 15 Emitting Countries (Latest Year)

Sector-level Emissions Analysis

Year-over-Year Growth Dynamics

🛠️ Tech Stack
Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Plotly, Streamlit

Data Source: IEA–EDGAR CO₂ dataset (1970–2023)  
Future Enhancements
Add scenario simulator for policy-driven projections.

Expand ML forecasting with XGBoost/LightGBM.

Deploy dashboard via Streamlit Cloud / Docker.
