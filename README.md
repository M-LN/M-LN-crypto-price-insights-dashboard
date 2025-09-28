# 🚀 Crypto Price Insights Dashboard

A professional cryptocurrency market analysis dashboard with machine learning capabilities. This portfolio project showcases real-time data processing, advanced analytics, and ML-powered price direction predictions.

## ✨ Key Features

### 📊 **Real-Time Analytics**
- Live cryptocurrency market data (Bitcoin, Ethereum, Cardano, Solana)
- 11 technical indicators (moving averages, volatility, momentum, drawdowns)
- Interactive price charts and performance metrics
- Professional web dashboard with Streamlit

### 🤖 **Machine Learning**
- Next-day price direction prediction (UP/DOWN)
- Gradient boosting classifier with cross-validation
- Feature importance analysis and confidence scores
- Model persistence and automatic training

### 🔧 **Production Ready**
- Comprehensive test suite (12/12 tests passing)
- GitHub Actions automation workflows
- Clean, modular architecture
- Professional error handling and caching

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/M-LN/crypto-price-insights-dashboard.git
cd crypto-price-insights-dashboard
pip install -r requirements.txt
```

### Run Dashboard
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

### Test ML Demo
```bash
python demo_ml.py
```

## 🎮 How to Use

1. **Select Cryptocurrency**: Choose Bitcoin, Ethereum, etc.
2. **Configure Timeframe**: Adjust days of historical data (7-90)
3. **Enable ML Predictions**: Check the ML checkbox in sidebar
4. **Generate Forecast**: Click "🎯 Generate ML Prediction"
5. **View Results**: See direction, confidence, and model performance
6. **Explore Analysis**: Check technical indicators and charts

## 🧪 Testing

```bash
python -m pytest tests/ -v
# ✅ 12/12 tests passing
```

## 📈 ML Performance

- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: ~50-55% (better than random)
- **Features**: 11 technical indicators
- **Validation**: 5-fold cross-validation
- **Output**: Direction (UP/DOWN) + Confidence %

## 🏗️ Architecture

```
├── src/crypto_dashboard/    # Core package
│   ├── api.py              # CoinGecko API client
│   ├── processing.py       # Feature engineering
│   ├── ml.py               # Machine learning
│   ├── insights.py         # Analytics
│   └── visuals.py          # Plotting
├── tests/                  # Test suite
├── models/                 # Trained ML models
├── .github/workflows/      # Automation
└── streamlit_app.py        # Web dashboard
```

## 🔄 Automation

- **Daily Data Pipeline**: Automated collection at 6 AM UTC
- **Model Training**: On-demand via GitHub Actions
- **Testing**: Continuous integration
- **Deployment**: Cloud-ready with Docker support

## 🌟 Portfolio Value

This project demonstrates:
- **Full-Stack ML**: API → Processing → ML → Web App
- **Production Practices**: Testing, automation, monitoring
- **Modern Tech Stack**: Python, scikit-learn, Streamlit
- **Professional Code**: Clean architecture, documentation

## 📊 Tech Stack

- **Backend**: Python, pandas, numpy, scikit-learn
- **Frontend**: Streamlit, plotly, matplotlib
- **API**: CoinGecko cryptocurrency data
- **ML**: Gradient boosting, feature engineering
- **Testing**: pytest with comprehensive coverage
- **Automation**: GitHub Actions workflows

## 🚀 Extension Ideas

- **Advanced ML**: LSTM, ensemble methods, sentiment analysis
- **Data Storage**: PostgreSQL, data lakes, real-time streaming
- **Deployment**: Azure/AWS hosting, load balancing
- **Features**: Portfolio optimization, risk management, alerts

---

**Built as a comprehensive data science portfolio project showcasing modern ML and web development practices.** 🎯