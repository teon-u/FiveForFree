# NASDAQ Predictor Frontend

A modern React application for real-time NASDAQ stock predictions using AI/ML models.

## Features

- **Real-time Updates**: WebSocket connection for live predictions every minute
- **Interactive Dashboard**: Card-based layout with Volume Top 100 and Gainers Top 100
- **Color-coded Predictions**:
  - Green (70%+ up probability)
  - Red (70%+ down probability)
  - Gray (< 70% probability)
- **Detailed Analysis**: Click any ticker to view:
  - Model comparison table (5 models)
  - 60-minute price chart
  - Hit rates based on 50-hour backtesting
- **Customizable Settings**: Adjust thresholds and filters
- **Responsive Design**: Works on desktop and mobile

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **Recharts** - Charts and visualizations
- **React Query** - Data fetching and caching
- **Zustand** - State management
- **Axios** - HTTP client

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Edit .env if needed to match your backend URL
```

### Development

```bash
# Start development server
npm run dev

# The app will be available at http://localhost:3000
```

### Build for Production

```bash
# Create optimized production build
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── Dashboard.jsx           # Main dashboard layout
│   │   ├── TickerCard.jsx          # Individual ticker card
│   │   ├── TickerGrid.jsx          # Grid of ticker cards
│   │   ├── PredictionPanel.jsx     # Detailed prediction modal
│   │   ├── ModelComparison.jsx     # Model comparison table
│   │   ├── PriceChart.jsx          # 60-minute price chart
│   │   └── SettingsPanel.jsx       # Settings sidebar
│   ├── hooks/               # Custom React hooks
│   │   ├── useWebSocket.js         # WebSocket connection
│   │   ├── usePredictions.js       # Predictions data
│   │   └── useModels.js            # Model data
│   ├── services/            # API services
│   │   └── api.js                  # Axios client
│   ├── stores/              # State management
│   │   └── settingsStore.js        # Settings store
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # React entry point
│   └── index.css            # Global styles
├── public/                  # Static assets
├── index.html               # HTML entry point
├── vite.config.js           # Vite configuration
├── tailwind.config.js       # Tailwind configuration
└── package.json             # Dependencies
```

## API Endpoints

The frontend expects the following backend endpoints:

- `GET /api/predictions?threshold=0.7` - Get predictions for both sections
- `GET /api/models/{ticker}` - Get model comparison data for a ticker
- `GET /api/prices/{ticker}?minutes=60` - Get price history for a ticker
- `WS /ws` - WebSocket for real-time updates

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000/ws
```

### Tailwind Colors

Custom colors are defined in `tailwind.config.js`:

- `strong-up`: #22c55e (80%+ up)
- `up`: #86efac (70-80% up)
- `strong-down`: #ef4444 (80%+ down)
- `down`: #fca5a5 (70-80% down)
- `neutral`: #e5e7eb (< 70%)

## Features in Detail

### WebSocket Updates

The application connects to the backend WebSocket and automatically refreshes data when:
- New predictions are available (every minute)
- Specific ticker data is updated
- Models are retrained

### Settings Panel

Customize your experience:
- **Target Percent**: 3-10% (expected price movement)
- **Probability Threshold**: 50-90% (minimum to highlight)
- **Direction Filter**: All / Up / Down

Settings are persisted in localStorage.

### Card Colors

Cards are color-coded based on probability and direction:
- Strong signals (80%+): Bold colors with glow effect
- Medium signals (70-80%): Lighter colors
- Weak signals (<70%): Gray/neutral

## Development

### Linting

```bash
npm run lint
```

### VS Code Setup

Recommended extensions:
- ESLint
- Tailwind CSS IntelliSense
- Prettier

## Troubleshooting

### WebSocket Not Connecting

1. Check that backend is running on `http://localhost:8000`
2. Verify WebSocket URL in `.env` file
3. Check browser console for connection errors

### API Errors

1. Verify backend is accessible at the configured URL
2. Check CORS settings on backend
3. Review network tab in browser DevTools

### Build Errors

1. Clear node_modules and reinstall: `rm -rf node_modules && npm install`
2. Clear Vite cache: `rm -rf node_modules/.vite`
3. Ensure Node.js version is 18 or higher

## License

MIT
