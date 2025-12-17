# Claude Code Custom Commands

This directory contains 9 specialized sub-agents (slash commands) for the NASDAQ Prediction System project.

## Key Metrics Update (2025-12-17)

**IMPORTANT: Use Precision instead of Accuracy/Hit Rate**
- **Precision** = TP / (TP + FP) - When model predicts UP, how often is it correct?
- Current best model Precision: ~37.5%
- Breakeven Precision for 3:1 R/R strategy: 30%
- Status: **PROFITABLE** with proper risk management

## Available Commands

### 1. `/model-test` - ML Model Testing
**Purpose:** Test and validate machine learning models
**Use Cases:**
- Verify all models are properly initialized and trained
- Test prediction accuracy and performance metrics
- Validate model serialization and loading
- Check ensemble model integration

**Example:**
```
/model-test
```

### 2. `/api-test` - API Endpoint Testing
**Purpose:** Test all FastAPI endpoints and API functionality
**Use Cases:**
- Test health checks and server status
- Validate all REST endpoints (tickers, predictions, models)
- Test WebSocket connections
- Verify response schemas and error handling

**Example:**
```
/api-test
```

### 3. `/data-check` - Data Quality Validation
**Purpose:** Check data collection and data quality
**Use Cases:**
- Validate database tables and schemas
- Check for data gaps or missing timestamps
- Test data collectors (Finnhub, Polygon)
- Verify feature engineering and label generation
- Check data freshness and collection schedule

**Example:**
```
/data-check
```

### 4. `/deploy-check` - Deployment Readiness
**Purpose:** Pre-deployment checklist and verification
**Use Cases:**
- Verify environment configuration
- Check all dependencies are installed
- Security audit (exposed keys, vulnerabilities)
- Test builds (frontend + backend)
- Verify database migrations
- Review production configuration

**Example:**
```
/deploy-check
```

### 5. `/debug-model` - Model Debugging
**Purpose:** Diagnose and fix ML model issues
**Use Cases:**
- Debug training failures
- Fix prediction errors
- Diagnose poor model performance
- Check memory and performance issues
- Analyze model-specific problems (LSTM, Transformer, etc.)

**Example:**
```
/debug-model

Specific issue: LSTM model failing to train on NVDA data
Error message: "RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long"
```

### 6. `/optimize` - Performance Optimization
**Purpose:** Identify and implement performance optimizations
**Use Cases:**
- Optimize database queries and indexes
- Improve API endpoint response times
- Speed up model inference
- Reduce frontend bundle size
- Optimize data collection and caching
- Reduce costs (API calls, compute)

**Example:**
```
/optimize

Focus on: API response times for /api/models/{ticker}/performance endpoint
Current latency: 2.5 seconds
```

### 7. `/code-review` - Code Quality Review
**Purpose:** Conduct thorough code reviews
**Use Cases:**
- Check code quality and best practices
- Identify security vulnerabilities
- Review architecture and design patterns
- Find performance issues and code smells
- Verify testing coverage
- Check documentation completeness

**Example:**
```
/code-review

Please review: src/models/ensemble_model.py
Focus on: error handling and performance
```

### 8. `/backtest` - Backtest Simulation (NEW)
**Purpose:** Run trading simulations and calculate expected returns
**Use Cases:**
- Simulate trading based on model predictions
- Calculate expected annual returns
- Test different risk/reward strategies
- Analyze win rate and drawdown
- Project profitability

**Example:**
```
/backtest

Strategy: 3:1 Reward/Risk
Target: +3%, Stop: -1%
Capital: $10,000
```

**Key Outputs:**
- Total trades and win rate
- Target hit vs stop loss ratio
- Total and average P&L
- Sharpe ratio and max drawdown
- Annual return projection

### 9. `/ui-test` - Frontend UI Testing (NEW)
**Purpose:** Test frontend UI using Playwright browser automation
**Use Cases:**
- Automated browser testing
- Test tab navigation and modals
- Verify ticker cards display correctly
- Test model detail and chart modals
- Check Precision displays (not Hit Rate!)

**Example:**
```
/ui-test

Focus: Test model detail modal for BIIB
Check: Precision values in ranking table
```

## How to Use

### Basic Usage
Simply type the command in Claude Code chat:
```
/model-test
```

### With Context
Provide additional context or specific areas to focus on:
```
/debug-model

Issue: Ensemble model predictions are all returning 0.5 probability
Affected tickers: NVDA, AAPL, MSFT
Recent changes: Updated meta learner weights calculation
```

### Combined Workflow
Use multiple commands in sequence:
```
1. /data-check  # First check if data is good
2. /model-test  # Then test if models work
3. /api-test    # Finally test API endpoints
```

## Command Categories

### 🧪 Testing & Validation
- `/model-test` - Test ML models (Precision-focused)
- `/api-test` - Test API endpoints
- `/data-check` - Validate data quality
- `/ui-test` - Frontend UI testing (Playwright)

### 📊 Trading & Analysis
- `/backtest` - Run trading simulations

### 🐛 Debugging & Troubleshooting
- `/debug-model` - Debug ML model issues

### 🚀 Deployment & Operations
- `/deploy-check` - Pre-deployment verification

### ⚡ Performance & Quality
- `/optimize` - Performance optimization
- `/code-review` - Code quality review

## Best Practices

1. **Use Before Commits:** Run `/code-review` before major commits
2. **Pre-Deployment:** Always run `/deploy-check` before deploying
3. **Regular Checks:** Run `/data-check` daily to ensure data quality
4. **Performance Monitoring:** Run `/optimize` weekly to identify bottlenecks
5. **After Changes:** Run `/model-test` after modifying ML code
6. **API Changes:** Run `/api-test` after modifying endpoints

## Tips

- **Be Specific:** Provide context and specific issues for better results
- **Sequential Use:** Use commands in logical order (data → model → api)
- **Save Reports:** Copy command outputs for documentation
- **Automate:** Consider running some checks in CI/CD pipeline
- **Custom Focus:** Tell the agent what to focus on for targeted analysis

## Examples of Effective Usage

### Scenario 1: New Feature Development
```
1. /code-review
   Focus: New financial tab implementation

2. /api-test
   Test: /api/models/{ticker}/financial endpoint

3. /deploy-check
   Verify: All dependencies for new feature
```

### Scenario 2: Performance Issue
```
1. /optimize
   Issue: Dashboard loading slowly
   Focus: Frontend bundle size and API latency

2. /api-test
   Measure: Response times for all endpoints
```

### Scenario 3: Production Incident
```
1. /debug-model
   Issue: Models returning NaN predictions

2. /data-check
   Verify: Recent data quality

3. /model-test
   Test: All model predictions
```

## Customization

You can modify these commands by editing the `.md` files in `.claude/commands/`:
- Add project-specific checks
- Customize checklists
- Add new commands
- Remove unnecessary sections

## Support

If a command isn't working as expected:
1. Check the command file syntax in `.claude/commands/`
2. Ensure the command description is clear
3. Provide more context when invoking the command
4. Verify you're using the latest version of Claude Code

---

**Last Updated:** 2025-12-17
**Project:** NASDAQ Short-Term Volatility Prediction System
**Total Commands:** 9

## Recent Updates

- **2025-12-17**: Added `/ui-test` command for Playwright browser testing
- **2025-12-17**: Added `/backtest` command for trading simulations
- **2025-12-17**: Updated `/model-test` to focus on Precision metrics
- **2025-12-17**: Changed from Hit Rate/Accuracy to Precision for meaningful model evaluation
