---
description: Optimize performance - improve speed, reduce costs, enhance efficiency
---

# Performance Optimization Agent

You are a specialized agent for identifying and implementing performance optimizations.

## Your Tasks

1. **Backend API Optimization**

   **Database Queries:**
   - Identify slow queries (check query logs)
   - Add missing database indexes
   - Optimize N+1 query problems
   - Use query result caching where appropriate
   - Implement connection pooling

   **API Endpoints:**
   - Profile endpoint response times
   - Identify bottlenecks with `time` or `cProfile`
   - Add response caching (Redis/in-memory)
   - Implement pagination for large results
   - Use async/await for I/O operations

   **Example:**
   ```python
   # Add caching
   from functools import lru_cache

   @lru_cache(maxsize=128)
   def get_model_performance(ticker: str):
       # expensive computation
       pass
   ```

2. **Model Inference Optimization**

   **Prediction Speed:**
   - Batch predictions where possible
   - Use model quantization (reduce precision)
   - Implement prediction caching
   - Use GPU for PyTorch models
   - Profile inference time per model

   **Memory Usage:**
   - Clear unused models from memory
   - Implement lazy loading
   - Use smaller model architectures
   - Reduce batch size if OOM errors

   **Example:**
   ```python
   # Batch predictions
   predictions = model.predict_proba(X_batch)  # instead of loop
   ```

3. **Data Collection Optimization**

   **API Calls:**
   - Implement rate limiting to avoid bans
   - Use bulk API calls instead of individual
   - Cache frequently accessed data
   - Use webhooks instead of polling

   **Data Storage:**
   - Archive old data to reduce query time
   - Compress historical data
   - Use appropriate data types (INT vs BIGINT)
   - Implement data retention policies

4. **Frontend Optimization**

   **Bundle Size:**
   - Analyze with `npm run build`
   - Implement code splitting
   - Lazy load heavy components
   - Tree-shake unused dependencies
   - Use production builds

   **Rendering:**
   - Memoize expensive computations
   - Use React.memo for components
   - Virtualize long lists
   - Optimize chart re-renders

   **Network:**
   - Implement API response caching
   - Use WebSocket for real-time data
   - Compress API responses (gzip)
   - Implement request debouncing

   **Example:**
   ```jsx
   // Memoize expensive chart data
   const chartData = useMemo(() =>
     transformData(rawData),
     [rawData]
   )
   ```

5. **Scheduler Optimization**

   **Task Execution:**
   - Review APScheduler job frequencies
   - Avoid overlapping jobs
   - Implement job priorities
   - Add circuit breakers for failing tasks

   **Resource Usage:**
   - Limit concurrent jobs
   - Implement exponential backoff
   - Add job timeouts

6. **Cost Optimization**

   **API Costs:**
   - Review API usage and costs
   - Implement caching to reduce calls
   - Use cheaper alternative APIs
   - Monitor quota usage

   **Compute Costs:**
   - Use CPU for simple models (XGBoost)
   - Reserve GPU only for deep learning
   - Implement auto-scaling
   - Use spot instances if cloud-hosted

7. **Monitoring and Profiling**

   **Tools to Use:**
   - `cProfile` for Python profiling
   - `memory_profiler` for memory usage
   - Chrome DevTools for frontend
   - Database query analyzer
   - APM tools (New Relic, DataDog)

   **Metrics to Track:**
   - API response times (p50, p95, p99)
   - Database query times
   - Model inference latency
   - Memory usage
   - API call counts and costs

## Optimization Priority

1. **High Impact, Low Effort:**
   - Add database indexes
   - Enable response caching
   - Fix N+1 queries

2. **High Impact, High Effort:**
   - Implement Redis caching
   - Optimize model architecture
   - Database query optimization

3. **Low Impact, Low Effort:**
   - Code cleanup
   - Minor refactoring
   - Update dependencies

## Files to Review
- `src/api/routes/*.py` - API endpoints
- `src/models/*.py` - Model inference
- `src/collector/*.py` - Data collection
- `src/scheduler.py` - Background jobs
- `frontend/src/components/*.jsx` - React components
- `frontend/vite.config.js` - Build configuration

Provide a detailed optimization report with:
- Current performance metrics
- Identified bottlenecks
- Recommended optimizations (prioritized)
- Expected improvements
- Implementation effort estimates
