# 📊 Crypto Price Insights Dashboard

A cryptocurrency price tracking and analytics dashboard built with Python.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

> **📌 Repository Setup Reminder:**
> - Add description: "Cryptocurrency price insights dashboard with real-time tracking and analytics"
> - Add topics: `python`, `cryptocurrency`, `bitcoin`, `dashboard`, `data-visualization`, `analytics`
> - Set these in GitHub Repository Settings → About section

## 🚀 Features

- 💹 Real-time cryptocurrency price tracking (Bitcoin, Ethereum, Cardano, Solana)
- 📈 Interactive price charts and visualizations with Plotly
- 📊 Advanced market analytics with 11 technical indicators
- 🤖 Machine learning-powered price direction predictions
- 📉 Historical data analysis with moving averages and volatility metrics
- 💰 Multi-currency support with CoinGecko API integration
- 🎯 Next-day price direction forecasting using gradient boosting
- ✅ Production-ready with comprehensive test coverage

## 🛠️ Technologies Used

This project leverages modern Python technologies and libraries:

- **Python 3.8+** - Core programming language
- **Streamlit** - Interactive web dashboard framework
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning models (Gradient Boosting Classifier)
- **Plotly & Matplotlib** - Data visualization and charting
- **requests** - API communication
- **CoinGecko API** - Real-time cryptocurrency market data
- **pytest** - Testing framework
- **GitHub Actions** - CI/CD automation

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/M-LN/M-LN-crypto-price-insights-dashboard.git
cd M-LN-crypto-price-insights-dashboard
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Running the Dashboard

Launch the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Using the Dashboard

1. **Select Cryptocurrency**: Choose from Bitcoin, Ethereum, Cardano, or Solana in the sidebar
2. **Configure Timeframe**: Adjust the slider to select days of historical data (7-90 days)
3. **View Analytics**: Explore real-time price charts and technical indicators
4. **Enable ML Predictions**: Check the "Enable ML Predictions" checkbox in the sidebar
5. **Generate Forecast**: Click "🎯 Generate ML Prediction" to get next-day price direction
6. **Explore Insights**: Review key insights, performance metrics, and technical analysis

### Running the ML Demo

Test the machine learning functionality standalone:

```bash
python demo_ml.py
```

### Running Tests

Execute the test suite to verify functionality:

```bash
python -m pytest tests/ -v
```

> **Note**: Add screenshots of the dashboard here when available to showcase the UI

## 📁 Project Structure

```
M-LN-crypto-price-insights-dashboard/
├── src/
│   └── crypto_dashboard/      # Core application package
│       ├── __init__.py
│       ├── api.py              # CoinGecko API client
│       ├── processing.py       # Data processing and feature engineering
│       ├── ml.py               # Machine learning models
│       ├── insights.py         # Analytics and insights generation
│       ├── visuals.py          # Plotting and visualization
│       └── pipeline.py         # Data pipeline orchestration
├── tests/                      # Test suite
│   ├── test_ml.py
│   └── test_processing.py
├── scripts/                    # Utility scripts
│   └── run_pipeline.py
├── models/                     # Trained ML models (persisted)
├── notebooks/                  # Jupyter notebooks for exploration
├── .github/
│   └── workflows/              # GitHub Actions automation
│       ├── daily-pipeline.yml
│       └── train-model.yml
├── streamlit_app.py            # Main web dashboard application
├── demo_ml.py                  # ML demonstration script
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## ⚙️ Configuration

### API Keys

This project uses the **CoinGecko API** which doesn't require an API key for basic usage. If you plan to make extensive API calls, consider:

- Setting up rate limiting
- Using the CoinGecko Pro API with authentication
- Implementing caching (already included in the dashboard)

### Customization

You can customize the dashboard by modifying:

- **Cryptocurrency list**: Edit the selectbox options in `streamlit_app.py`
- **Technical indicators**: Adjust calculations in `src/crypto_dashboard/processing.py`
- **ML model parameters**: Modify hyperparameters in `src/crypto_dashboard/ml.py`
- **Visualization styles**: Update chart configurations in `streamlit_app.py` and `src/crypto_dashboard/visuals.py`

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `python -m pytest tests/ -v`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Contribution Guidelines

- Write clear, descriptive commit messages
- Add tests for new features
- Update documentation as needed
- Follow existing code style and conventions
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**M-LN** - [@M-LN](https://github.com/M-LN)

Project Link: [https://github.com/M-LN/M-LN-crypto-price-insights-dashboard](https://github.com/M-LN/M-LN-crypto-price-insights-dashboard)

---

**Built with ❤️ using Python, Streamlit, and Machine Learning** 🚀