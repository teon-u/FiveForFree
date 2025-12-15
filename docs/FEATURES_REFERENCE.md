# Feature Engineering Quick Reference

## All 57 Features by Category

### 📊 Price-Based Features (15)

| Feature | Description | TA-Lib | Normalized |
|---------|-------------|--------|------------|
| `returns_1m` | 1-minute return | ✗ | ✓ |
| `returns_5m` | 5-minute return | ✗ | ✓ |
| `returns_15m` | 15-minute return | ✗ | ✓ |
| `returns_30m` | 30-minute return | ✗ | ✓ |
| `returns_60m` | 60-minute return | ✗ | ✓ |
| `ma_5` | 5-period SMA | ✓ | ✗ |
| `ma_15` | 15-period SMA | ✓ | ✗ |
| `ma_60` | 60-period SMA | ✓ | ✗ |
| `price_vs_ma_5` | Price deviation from MA5 | ✗ | ✓ |
| `price_vs_ma_15` | Price deviation from MA15 | ✗ | ✓ |
| `price_vs_ma_60` | Price deviation from MA60 | ✗ | ✓ |
| `ma_cross_5_15` | MA5/MA15 crossover signal | ✗ | ✓ |
| `price_vs_vwap` | Price deviation from VWAP | ✗ | ✓ |
| `price_momentum_5m` | 5-minute momentum | ✗ | ✓ |
| `price_momentum_15m` | 15-minute momentum | ✗ | ✓ |

### 📉 Volatility-Based Features (10)

| Feature | Description | TA-Lib | Normalized |
|---------|-------------|--------|------------|
| `atr_14_normalized` | Average True Range (14) | ✓ | ✓ |
| `bb_position` | Position within Bollinger Bands | ✓ | ✓ |
| `bb_width` | Bollinger Band width | ✓ | ✓ |
| `volatility_5m` | 5-minute volatility (std) | ✗ | ✓ |
| `volatility_15m` | 15-minute volatility (std) | ✗ | ✓ |
| `volatility_60m` | 60-minute volatility (std) | ✗ | ✓ |
| `price_acceleration` | Price acceleration (2nd deriv) | ✗ | ✓ |
| `high_low_range` | High-low range | ✗ | ✓ |
| `intraday_range` | Intraday range from open | ✗ | ✓ |
| `volatility_ratio` | Short/long volatility ratio | ✗ | ✓ |

### 📈 Volume-Based Features (8)

| Feature | Description | TA-Lib | Normalized |
|---------|-------------|--------|------------|
| `volume_ratio` | Current/average volume | ✗ | ✓ |
| `volume_ma_5` | 5-period volume MA | ✓ | ✗ |
| `volume_ma_15` | 15-period volume MA | ✓ | ✗ |
| `volume_trend` | Volume trend indicator | ✗ | ✓ |
| `obv_normalized` | On-Balance Volume | ✓ | ✓ |
| `money_flow_ratio` | Money flow ratio | ✗ | ✓ |
| `mfi_14` | Money Flow Index (14) | ✓ | ✗ |
| `vpt_cumsum` | Volume Price Trend | ✗ | ✗ |

### 📖 Order Book Features (8)

| Feature | Description | Source | Normalized |
|---------|-------------|--------|------------|
| `bid_ask_spread` | Absolute bid-ask spread | L2 | ✗ |
| `spread_pct` | Spread as % of mid price | L2 | ✓ |
| `imbalance` | Bid-ask volume imbalance | L2 | ✓ |
| `bid_depth` | Total bid volume | L2 | ✗ |
| `ask_depth` | Total ask volume | L2 | ✗ |
| `depth_ratio` | Bid/ask depth ratio | L2 | ✓ |
| `depth_weighted_mid_price` | Depth-weighted price | L2 | ✗ |
| `order_flow_imbalance` | Order flow imbalance | L2 | ✓ |

### 🚀 Momentum Features (8)

| Feature | Description | TA-Lib | Range |
|---------|-------------|--------|-------|
| `rsi_14` | Relative Strength Index | ✓ | 0-100 |
| `macd` | MACD line | ✓ | ℝ |
| `macd_signal` | MACD signal line | ✓ | ℝ |
| `macd_hist` | MACD histogram | ✓ | ℝ |
| `stoch_k` | Stochastic %K | ✓ | 0-100 |
| `stoch_d` | Stochastic %D | ✓ | 0-100 |
| `williams_r` | Williams %R | ✓ | -100-0 |
| `cci_14` | Commodity Channel Index | ✓ | ℝ |

### 🌍 Market Context Features (5)

| Feature | Description | Source | Normalized |
|---------|-------------|--------|------------|
| `spy_return` | S&P 500 return | SPY | ✓ |
| `qqq_return` | NASDAQ 100 return | QQQ | ✓ |
| `vix_level` | VIX volatility index | VIX | ✓ |
| `sector_etf_return` | Sector ETF return | Sector ETF | ✓ |
| `market_correlation` | Market correlation | Computed | ✓ |

### ⏰ Time-Based Features (3)

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `minutes_since_open` | Minutes since 9:30 AM | Integer | 0-390 |
| `day_of_week` | Day of week | Integer | 0-4 |
| `is_option_expiry` | Option expiry day | Binary | 0-1 |

---

## Feature Engineering Pipeline

```python
from src.processor import FeatureEngineer

# Initialize
fe = FeatureEngineer()

# Input: OHLCV DataFrame
minute_bars = pd.DataFrame({
    'timestamp': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Optional: Order book data
order_book = {
    'bids': [(price, size), ...],
    'asks': [(price, size), ...],
    'bid_total_volume': float,
    'ask_total_volume': float,
    'imbalance': float
}

# Optional: Market context
market_data = {
    'spy_return': float,
    'qqq_return': float,
    'vix_level': float,
    'sector_return': float,
    'correlation': float
}

# Compute all 57 features
df_features = fe.compute_features(
    df=minute_bars,
    order_book=order_book,     # Optional
    market_data=market_data    # Optional
)

# Get feature names
feature_names = fe.get_feature_names()  # List of 57 features
```

---

## Label Generation Pipeline

```python
from src.processor import LabelGenerator

# Initialize
lg = LabelGenerator(
    target_percent=5.0,              # 5% target
    prediction_horizon_minutes=60,    # 60-minute horizon
    commission_pct=0.1                # 0.1% commission
)

# Generate labels (vectorized - fast)
df_labels = lg.generate_labels_vectorized(minute_bars)

# Output columns:
# - label_up: 1 if +5% reached within 60 min
# - label_down: 1 if -5% reached within 60 min
# - max_gain: Maximum gain percentage
# - max_loss: Maximum loss percentage

# Get statistics
stats = lg.get_label_statistics(df_labels)
print(f"Up rate: {stats['up_rate']:.1%}")
print(f"Down rate: {stats['down_rate']:.1%}")
```

---

## Feature Categories by Importance

### High-Frequency Trading Focus
1. **Order Book** (8) - Level 2 data, spreads, imbalances
2. **Price-Based** (15) - Returns, MAs, momentum
3. **Volatility** (10) - ATR, BB, ranges

### Context & Confirmation
4. **Momentum** (8) - RSI, MACD, Stochastic
5. **Volume** (8) - Flow, OBV, MFI

### Market Environment
6. **Market Context** (5) - SPY, QQQ, VIX
7. **Time** (3) - Session timing, expiry

---

## Feature Value Ranges

### Bounded Features (0-1 normalized)
- All returns and price deviations
- `bb_position`, `spread_pct`, `vix_level`
- All ratios and imbalances
- `is_option_expiry`

### Bounded Features (specific ranges)
- `rsi_14`: 0-100
- `stoch_k`, `stoch_d`: 0-100
- `williams_r`: -100 to 0
- `mfi_14`: 0-100
- `day_of_week`: 0-4
- `minutes_since_open`: 0-390

### Unbounded Features
- Moving averages (price scale)
- MACD components
- CCI
- Volume measures (share scale)

---

## Missing Data Handling

### Strategy
1. **Forward Fill** - Use previous valid value
2. **Backward Fill** - Use next valid value (for start of series)
3. **Zero Fill** - Final fallback

### Graceful Degradation
- **No order book data**: All order book features = default values
- **No market data**: All market context features = default values
- **Missing VWAP**: Computed from price × volume

---

## Performance Characteristics

### Computation Speed (estimated)
- **1000 rows**: ~100ms
- **10000 rows**: ~500ms
- **100000 rows**: ~3s

### Memory Usage
- Base DataFrame: ~1 MB per 10K rows
- With features: ~5 MB per 10K rows
- Overhead: ~5x

### Bottlenecks
- TA-Lib functions (optimized C code)
- Rolling window operations
- Label generation (future scanning)

---

## Integration Points

### Input Sources
- **Polygon.io**: Minute bars, Level 2 quotes
- **Market Data**: SPY/QQQ/VIX prices
- **Time Data**: System clock

### Output Consumers
- **Model Training**: XGBoost, LightGBM, LSTM, Transformer
- **Real-time Prediction**: Live feature computation
- **Backtesting**: Historical label validation

---

## Validation Checklist

- [ ] All 57 features computed without errors
- [ ] No missing values in output (all filled)
- [ ] Feature ranges are reasonable (no infinities/NaNs)
- [ ] Labels match expected distribution (5-30% positive rate)
- [ ] No data leakage (features use only past data)
- [ ] Performance meets requirements (<1s per 10K rows)

---

**Last Updated**: 2024-12-15
**Version**: 1.0
**Status**: Production Ready
