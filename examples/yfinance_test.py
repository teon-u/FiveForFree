#!/usr/bin/env python3
"""
Yahoo Finance 데이터 수집 테스트
API 키 불필요, 완전 무료!
"""

import yfinance as yf
from datetime import datetime, timedelta

print("=== Yahoo Finance 테스트 ===\n")

# 1. 실시간 Quote
print("1. AAPL 실시간 데이터:")
ticker = yf.Ticker("AAPL")
info = ticker.info

print(f"   회사명: {info.get('longName', 'N/A')}")
print(f"   현재가: ${info.get('currentPrice', info.get('regularMarketPrice', 0)):.2f}")
print(f"   거래량: {info.get('volume', 0):,}")
print(f"   시가총액: ${info.get('marketCap', 0):,}")
print()

# 2. 1분봉 데이터 (최근 7일)
print("2. AAPL 1분봉 데이터 (최근 2일):")
df_1m = ticker.history(period='2d', interval='1m')

if not df_1m.empty:
    print(f"   데이터 포인트: {len(df_1m)}개")
    print(f"   최신 종가: ${df_1m['Close'].iloc[-1]:.2f}")
    print(f"   최고가: ${df_1m['High'].max():.2f}")
    print(f"   최저가: ${df_1m['Low'].min():.2f}")
    print(f"   총 거래량: {df_1m['Volume'].sum():,}")
    print()
    print("   최근 5개 데이터:")
    print(df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].tail())
else:
    print("   ⚠ 데이터 없음 (장 마감 시간이거나 주말)")
print()

# 3. 5분봉 데이터 (최근 60일)
print("3. AAPL 5분봉 데이터 (최근 5일):")
df_5m = ticker.history(period='5d', interval='5m')

if not df_5m.empty:
    print(f"   데이터 포인트: {len(df_5m)}개")
    print(f"   날짜 범위: {df_5m.index[0]} ~ {df_5m.index[-1]}")
print()

# 4. 여러 종목 동시 조회
print("4. 인기 종목 동시 조회:")
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
data = yf.download(symbols, period='1d', interval='1d', progress=False)

if 'Close' in data.columns:
    for symbol in symbols:
        try:
            price = data['Close'][symbol].iloc[-1]
            print(f"   {symbol}: ${price:.2f}")
        except:
            print(f"   {symbol}: 데이터 없음")
print()

print("✅ Yahoo Finance 테스트 완료!")
print("\n장점:")
print("- ✅ 완전 무료, API 키 불필요")
print("- ✅ 1분봉 데이터 (최근 7일)")
print("- ✅ 5분봉 데이터 (최근 60일)")
print("- ✅ 실시간 데이터 (15-20분 지연 가능)")
print("\n단점:")
print("- ⚠ 간헐적 rate limiting (과도한 요청 시)")
print("- ⚠ 1분봉은 최근 7일로 제한")
