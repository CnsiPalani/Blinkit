

<div align="center">
	<img src="favicon-96x96.png" width="72" style="border-radius: 16px; box-shadow: 0 2px 8px #b2dfdb;" alt="Blinkit Logo"/>
</div>

# AI-Powered Blinkit Business Decision Platform


An end-to-end, modular project that unifies Marketing, Sales, and Operations data to:
- Calculate daily **ROAS** (Return on Ad Spend)
- Visualize trends in a beautiful, interactive **Streamlit** dashboard
- Predict delivery delay **risk** with a classification model (AUC target > 0.80)
- Optional **RAG** chatbot to summarize root causes from customer feedback

<div align="center">
	<img src="https://img.shields.io/badge/streamlit-dashboard-green?logo=streamlit"/>
	<img src="https://img.shields.io/badge/python-3.9+-blue?logo=python"/>
	<img src="https://img.shields.io/badge/ML-AUC%20%3E%200.80-success?logo=scikit-learn"/>
</div>


## Features & Look
- 🛒 Modern sidebar with grocery and navigation icons
- 💡 Stylish header and subtitle for business context
- 📊 KPI cards with icons, gradients, and hover effects
- 📈 Dual-axis charts for Revenue vs Spend
- 🛠️ Model training and risk calculator with tooltips
- 🤖 RAG assistant (optional, for customer feedback Q&A)

## Repo Structure
```
sql/roas_analysis.sql           # PostgreSQL CTEs to build daily master view
src/config.py                   # Config & paths
src/utils.py                    # Helper functions
src/data_engineering.py         # Build master_daily_view.csv
src/ml_train.py                 # Train delay risk model and save models/delay_risk_model.pkl
app.py                          # Streamlit app (ROI + Risk Calculator + RAG)
models/                         # Saved models
requirements.txt                # Python deps
README.md                       # This guide
favicon-96x96.png               # App and sidebar icon
```

## Data Expectations
Place CSVs in `data/` (or set env vars):
- `blinkit_marketing_performance.csv` with columns: `date, channel, spend, impressions`
- `blinkit_orders.csv` with columns: `order_id, order_date, order_total, promised_time, actual_time, region, [optional] feedback`


## Getting Started
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt

# 1) Build master view
python src/data_engineering.py

# 2) Train delay risk model
python src/ml_train.py

# 3) Run dashboard
streamlit run app.py
```


## Notes
- ROAS is computed daily by full outer joining marketing spend and aggregated order revenue.
- Days with zero spend yield `ROAS = NULL` (to avoid infinity). Days with zero revenue yield `ROAS = 0`.
- The ML model uses `hour_of_day`, `day_of_week`, and `region`. Adjust `BLINKIT_LATE_THRESHOLD_MIN` if your SLA differs.
- RAG features are optional and require HuggingFace Transformers/SentenceTransformers. Provide a `feedback` column for richer analysis.

## Environment Variables (optional)
- `BLINKIT_DATA_DIR` (default: `data`)
- `BLINKIT_MARKETING_CSV`, `BLINKIT_ORDERS_CSV`, `BLINKIT_MASTER_CSV`
- `BLINKIT_MODEL_PATH` (default: `models/delay_risk_model.pkl`)
- `BLINKIT_LATE_THRESHOLD_MIN` (default: `10` minutes)
- `OPENAI_API_KEY` for RAG


## Evaluation & Targets
- Data Logic: Correct date-based join for ROAS.
- Model Accuracy: Aim **AUC > 0.80** (depends on data quality).
- Business Impact: Dashboard highlights profitable vs loss-making days via dual-axis chart.

---
<div align="center" style="color:#009e60; font-size:1.1rem; margin-top:2em;">
Made with <b>Streamlit</b> | <b>Blinkit IQ</b> &copy; 2025
</div>
- Code Modularity: SQL, ETL, ML, and UI are separated.

---

Built for quick commerce decisioning and aligned with the Blinkit final project brief.
