# NASDAQ Prediction API - Examples

## Starting the API

```bash
# Development mode with auto-reload
python run_api.py --reload

# Production mode
python run_api.py --host 0.0.0.0 --port 8000 --workers 4

# Or using uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- WebSocket: ws://localhost:8000/ws

## REST API Examples

### Health Check

```bash
# Basic health check
curl http://localhost:8000/api/health

# Readiness check
curl http://localhost:8000/api/health/ready

# Liveness check
curl http://localhost:8000/api/health/live
```

### Tickers

```bash
# List all tickers
curl http://localhost:8000/api/tickers

# List with pagination
curl "http://localhost:8000/api/tickers?limit=10&offset=0"

# Get specific ticker details
curl http://localhost:8000/api/tickers/AAPL
```

### Predictions

```bash
# Get prediction for single ticker
curl http://localhost:8000/api/predictions/AAPL

# Get prediction with all models
curl "http://localhost:8000/api/predictions/AAPL?include_all_models=true"

# Get prediction with market context
curl "http://localhost:8000/api/predictions/AAPL?include_context=true"

# Batch predictions
curl -X POST http://localhost:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "GOOGL", "MSFT"],
    "include_all_models": false
  }'

# Get top opportunities
curl "http://localhost:8000/api/predictions/top/opportunities?tickers=AAPL&tickers=GOOGL&tickers=MSFT&direction=up&top_n=5"
```

### Models

```bash
# Get model performance for specific ticker
curl http://localhost:8000/api/models/AAPL

# Get all models performance
curl http://localhost:8000/api/models

# Get prediction statistics
curl http://localhost:8000/api/models/AAPL/stats

# Get overall summary
curl http://localhost:8000/api/models/summary/overall
```

## WebSocket Examples

### Python Client

```python
import asyncio
import json
import websockets

async def connect_to_predictions():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        message = await websocket.recv()
        print(f"Connected: {message}")

        # Subscribe to tickers
        subscribe_msg = {
            "type": "subscribe",
            "tickers": ["AAPL", "GOOGL", "MSFT"]
        }
        await websocket.send(json.dumps(subscribe_msg))

        # Request immediate prediction
        predict_msg = {
            "type": "predict",
            "ticker": "AAPL"
        }
        await websocket.send(json.dumps(predict_msg))

        # Listen for updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Received: {data['type']}")

            if data['type'] == 'prediction_update':
                print(f"Ticker: {data['ticker']}")
                print(f"Signal: {data['trading_signal']}")
                print(f"UP: {data['up_probability']:.2%}")
                print(f"DOWN: {data['down_probability']:.2%}")

asyncio.run(connect_to_predictions())
```

### JavaScript Client

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
    console.log('Connected to WebSocket');

    // Subscribe to tickers
    ws.send(JSON.stringify({
        type: 'subscribe',
        tickers: ['AAPL', 'GOOGL', 'MSFT']
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Message type:', data.type);

    if (data.type === 'prediction_update') {
        console.log(`${data.ticker}: ${data.trading_signal}`);
        console.log(`UP: ${(data.up_probability * 100).toFixed(2)}%`);
        console.log(`DOWN: ${(data.down_probability * 100).toFixed(2)}%`);
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('WebSocket disconnected');
};

// Ping every 30 seconds to keep connection alive
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000);
```

## WebSocket Message Types

### Client → Server

#### Subscribe
```json
{
    "type": "subscribe",
    "tickers": ["AAPL", "GOOGL"]
}
```

#### Unsubscribe
```json
{
    "type": "unsubscribe",
    "tickers": ["AAPL"]
}
```

#### Request Prediction
```json
{
    "type": "predict",
    "ticker": "AAPL"
}
```

#### Ping (keepalive)
```json
{
    "type": "ping"
}
```

### Server → Client

#### Connected
```json
{
    "type": "connected",
    "message": "Connected to NASDAQ Prediction System",
    "timestamp": "2025-12-15T03:00:00.000Z"
}
```

#### Prediction Update
```json
{
    "type": "prediction_update",
    "ticker": "AAPL",
    "timestamp": "2025-12-15T03:01:00.000Z",
    "current_price": 185.50,
    "up_probability": 0.75,
    "down_probability": 0.25,
    "best_up_model": "xgboost",
    "best_down_model": "lightgbm",
    "up_model_accuracy": 0.68,
    "down_model_accuracy": 0.72,
    "trading_signal": "BUY",
    "confidence_level": "HIGH"
}
```

#### Heartbeat
```json
{
    "type": "heartbeat",
    "timestamp": "2025-12-15T03:01:30.000Z",
    "connections": 5
}
```

#### Error
```json
{
    "type": "error",
    "message": "Error description"
}
```

## Response Models

### PredictionResponse
- `ticker`: Stock symbol
- `timestamp`: Prediction timestamp (ISO 8601)
- `current_price`: Current stock price
- `up_probability`: Probability of TARGET_PERCENT%+ upward movement (0-1), currently 1%
- `down_probability`: Probability of TARGET_PERCENT%+ downward movement (0-1), currently 1%
- `best_up_model`: Best performing model for up prediction
- `best_down_model`: Best performing model for down prediction
- `up_model_accuracy`: 50-hour accuracy of up model (0-1)
- `down_model_accuracy`: 50-hour accuracy of down model (0-1)
- `trading_signal`: "BUY", "SELL", or "HOLD"
- `confidence_level`: "VERY_HIGH", "HIGH", "MEDIUM", or "LOW"

### Trading Signal Thresholds
- **BUY**: `up_probability >= settings.PROBABILITY_THRESHOLD` (default 0.70)
- **SELL**: `down_probability >= settings.PROBABILITY_THRESHOLD` (default 0.70)
- **HOLD**: Neither threshold met

### Confidence Levels
- **VERY_HIGH**: probability >= 0.80
- **HIGH**: probability >= 0.70
- **MEDIUM**: probability >= 0.60
- **LOW**: probability < 0.60

## Frontend Integration

### React Example

```typescript
import { useEffect, useState } from 'react';

interface Prediction {
  ticker: string;
  up_probability: number;
  down_probability: number;
  trading_signal: string;
  confidence_level: string;
  current_price: number;
}

function PredictionsDashboard() {
  const [predictions, setPredictions] = useState<Map<string, Prediction>>(new Map());
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    const websocket = new WebSocket('ws://localhost:8000/ws');

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'prediction_update') {
        setPredictions(prev => new Map(prev).set(data.ticker, data));
      }
    };

    websocket.onopen = () => {
      // Subscribe to tickers
      websocket.send(JSON.stringify({
        type: 'subscribe',
        tickers: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
      }));
    };

    setWs(websocket);

    return () => websocket.close();
  }, []);

  return (
    <div>
      {Array.from(predictions.values()).map(pred => (
        <div key={pred.ticker}>
          <h3>{pred.ticker}</h3>
          <p>Signal: {pred.trading_signal}</p>
          <p>UP: {(pred.up_probability * 100).toFixed(2)}%</p>
          <p>DOWN: {(pred.down_probability * 100).toFixed(2)}%</p>
        </div>
      ))}
    </div>
  );
}
```

## Error Handling

All endpoints return standard HTTP status codes:
- `200`: Success
- `404`: Resource not found (ticker, model, etc.)
- `422`: Validation error
- `500`: Internal server error
- `503`: Service unavailable (models not loaded)

Error response format:
```json
{
  "detail": "Error description"
}
```
