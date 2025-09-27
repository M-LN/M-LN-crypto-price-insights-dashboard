# Crypto Price Insights Dashboard

A portfolio-ready data analysis case study that demonstrates how to ingest cryptocurrency pricing data from a public API, engineer meaningful features, surface actionable insights, and present them in a lightweight dashboard.

## Project Highlights
- **API integration** with CoinGecko's public market data.
- **Data wrangling & feature engineering** for returns, rolling statistics, and volatility.
- **Insight generation** with narrative-ready summaries.
- **Visual analytics** using Matplotlib and Plotly.
- **Dashboard prototype** powered by Streamlit.
- **Tested, reusable pipeline** designed for extension into ML experiments or automation.

## Roadmap Overview
1. **Ingest**: Pull configurable historical price data for any supported coin.
2. **Prepare**: Clean the time series, add returns, rolling metrics, and volatility measures.
3. **Analyze**: Summarize performance and volatility, surface weekend/weekday patterns.
4. **Communicate**: Render charts, tabular highlights, and narrative insights.
5. **Extend** (optional): Add ML classifiers, automate daily refresh, or persist results.

## Repository Structure
```
crypto_price_insights_dashboard/
├── data/                    # Optional local exports (CSV, cache)
├── notebooks/               # Exploratory notebooks (starter template provided)
├── src/
│   └── crypto_dashboard/
│       ├── __init__.py      # Package exports
│       ├── api.py           # CoinGecko API client helpers
│       ├── processing.py    # Feature engineering & statistics
│       ├── insights.py      # Narrative insight generation
│       ├── visuals.py       # Plotly/Matplotlib chart helpers
│       └── pipeline.py      # Orchestration helpers
├── streamlit_app.py         # Streamlit dashboard prototype
├── tests/
│   └── test_processing.py   # Unit tests for feature generation
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation (this file)
```

## Quickstart
```powershell
# 1. (Optional) create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the data pipeline for Bitcoin (saves CSV in ./data)
python -m crypto_dashboard.pipeline --coin-id bitcoin --days 60 --export data/btc_prices.csv

# (Alternative) Run the CLI wrapper
python scripts/run_pipeline.py --coin-id ethereum --days 120

# 4. Start the Streamlit dashboard
streamlit run streamlit_app.py
```

## Notebook Template
A starter notebook is available at `notebooks/starter_analysis.ipynb` to capture additional exploratory work or visuals tailored for your portfolio.

## Extending the Case Study
- **Machine Learning**: Train a light gradient boosting classifier to predict next-day direction using engineered features.
- **Automation**: Schedule the pipeline with GitHub Actions or a cron job to keep insights fresh.
- **Deployment**: Host the Streamlit app on Streamlit Community Cloud or Azure Web Apps.
- **Data Storage**: Persist daily snapshots to a Postgres instance or a data lake for historical tracking.

## License
MIT License. Built for educational portfolio use.
