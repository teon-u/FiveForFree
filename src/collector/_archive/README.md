# Archived Files

This directory contains code that was developed but is no longer actively used.

## polygon_client.py

**Archived Date**: 2024-12-21

**Reason**: Polygon.io API integration was replaced by Yahoo Finance for minute bar data collection.

**Context**:
- Polygon.io free tier has limitations that made it less suitable for our use case
- Yahoo Finance provides unlimited free access to OHLCV data
- See `FINNHUB_MIGRATION.md` in project root for full migration details

**If Reactivating**:
1. Add `POLYGON_API_KEY` to `config/settings.py`
2. Move file back to `src/collector/`
3. Export from `src/collector/__init__.py`
4. Update dependencies.py to use the Polygon client
