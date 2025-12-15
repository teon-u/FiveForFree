# NASDAQ Short-Term Volatility Prediction System

AI-powered system that predicts the probability of NASDAQ stocks experiencing 5%+ price movements within one hour.

## 🎯 Overview

- **Target**: High-volatility NASDAQ stocks (Top 100 by volume + Top 100 gainers)
- **Prediction**: Probability of 5%+ up/down movement in next 60 minutes
- **Models**: 5 ML models per ticker (XGBoost, LightGBM, LSTM, Transformer, Ensemble)
- **Data Source**: Polygon.io Developer Plan ($79/month)
- **Hardware**: RTX 5080 GPU, AMD Ryzen 9800X3D, 64GB RAM

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.10+
python --version

# CUDA 12.0+ for GPU support
nvidia-smi
```

### 2. Installation

```bash
# Clone repository
git clone https://github.com/teon-u/FiveForFree.git
cd FiveForFree

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Setup frontend
cd frontend
npm install
cd ..
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Polygon.io API key
nano .env
```

### 4. Initial Setup

```bash
# Initialize database
python scripts/init_database.py

# Collect historical data (30 days)
python scripts/collect_historical.py 30

# Train initial models
python scripts/train_all_models.py
```

### 5. Run System

```bash
# Terminal 1: Start backend
python scripts/run_system.py

# Terminal 2: Start frontend dev server
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## 📁 Project Structure

```
FiveForFree/
├── config/              # Configuration files
├── src/
│   ├── collector/       # Data collection from Polygon.io
│   ├── processor/       # Feature engineering & labeling
│   ├── models/          # ML models (XGBoost, LSTM, etc.)
│   ├── trainer/         # GPU-accelerated training
│   ├── predictor/       # Real-time predictions
│   ├── backtester/      # Performance simulation
│   ├── api/             # FastAPI backend
│   └── utils/           # Utilities
├── frontend/            # React + Vite + Tailwind UI
├── data/                # Raw & processed data
├── scripts/             # Automation scripts
└── tests/               # Test suites
```

## 🎨 Tech Stack

### Backend
- **API**: FastAPI + WebSocket
- **ML**: XGBoost, LightGBM, PyTorch (LSTM/Transformer)
- **Data**: Polygon.io API, Pandas, NumPy
- **Database**: SQLite (SQLAlchemy)
- **Scheduler**: APScheduler

### Frontend
- **Framework**: React 18 + Vite
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **State**: React Query + Zustand
- **WebSocket**: native WebSocket API

## 📊 Features

### Data Collection
- ✅ Hourly ticker selection (top volume + gainers)
- ✅ Per-minute OHLCV bars + VWAP
- ✅ Level 2 order book (bid/ask imbalance)
- ✅ Market context (SPY, QQQ, VIX, sectors)

### Feature Engineering
- 📈 57 engineered features across 7 categories
- 🎯 Automatic label generation (5% threshold)
- ⚡ GPU-accelerated processing

### Machine Learning
- 🤖 5 models per ticker per direction (up/down)
- 🏆 Automatic best-model selection (50-hour accuracy)
- 🔄 Incremental learning (hourly)
- 📊 Full retraining (daily after market close)

### Backtesting
- 📉 50-hour rolling window simulation
- 💰 "5% OR 1 hour" liquidation rule
- 📈 Per-model hit rate tracking

### Real-time UI
- 🎴 Card-based ticker display (volume/gainers)
- 🟢🔴 Color-coded probabilities
- 📊 Model performance dashboard
- 📈 60-minute price charts
- 🔄 WebSocket live updates

## ⚙️ Automation Workflow

### GitHub Actions
- ✅ Automated testing on push/PR
- ✅ Code quality checks (Black, Flake8)
- ✅ Frontend build verification
- ✅ Docker image builds

### Continuous Learning
- 🔄 Hourly: Incremental training
- 📅 Daily: Full model retraining
- 🎯 Auto: Best-model selection

## 📈 Performance Metrics

| Scenario | Accuracy | Monthly Trades | Expected Return |
|----------|----------|----------------|-----------------|
| Optimistic | 75% | 200 | +30% |
| Realistic | 65% | 100 | +10% |
| Pessimistic | 55% | 50 | ±0% |

## 🛠️ Development

```bash
# Run tests
pytest

# Format code
black .

# Lint code
flake8 src/

# Frontend development
cd frontend
npm run dev        # Dev server
npm run build      # Production build
npm run preview    # Preview build
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. Not financial advice. Trading involves substantial risk of loss. Always do your own research and never invest more than you can afford to lose.

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/teon-u/FiveForFree/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/teon-u/FiveForFree/discussions)
