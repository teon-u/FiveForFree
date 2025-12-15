#!/usr/bin/env python3
"""
간단한 Finnhub API 사용 예제
로컬 환경에서 직접 실행해보세요.
"""

import finnhub

# API 키 설정 (여기에 직접 입력)
API_KEY = "d4vpv11r01qs25f1ls5gd4vpv11r01qs25f1ls60"

# Client 생성
client = finnhub.Client(api_key=API_KEY)

print("=== Finnhub API 테스트 ===\n")

# 1. AAPL Quote
print("1. AAPL 실시간 Quote:")
quote = client.quote('AAPL')
print(f"   현재가: ${quote['c']:.2f}")
print(f"   고가: ${quote['h']:.2f}")
print(f"   저가: ${quote['l']:.2f}")
print(f"   전일 종가: ${quote['pc']:.2f}")
print()

# 2. 여러 종목 조회
print("2. 인기 종목 조회:")
symbols = ['MSFT', 'GOOGL', 'TSLA', 'NVDA']
for symbol in symbols:
    q = client.quote(symbol)
    change = ((q['c'] - q['pc']) / q['pc'] * 100) if q['pc'] > 0 else 0
    print(f"   {symbol}: ${q['c']:.2f} ({change:+.2f}%)")
print()

# 3. 5분봉 캔들 데이터
print("3. AAPL 5분봉 데이터 (최근 24시간):")
from datetime import datetime, timedelta

to_ts = int(datetime.now().timestamp())
from_ts = int((datetime.now() - timedelta(hours=24)).timestamp())

candles = client.stock_candles('AAPL', '5', from_ts, to_ts)

if candles['s'] == 'ok':
    print(f"   캔들 개수: {len(candles['c'])}개")
    print(f"   최신 종가: ${candles['c'][-1]:.2f}")
    print(f"   최고가: ${max(candles['h']):.2f}")
    print(f"   최저가: ${min(candles['l']):.2f}")
print()

print("✅ 테스트 완료!")
print("\n참고:")
print("- 무료 티어: 60 API calls/minute")
print("- 5분봉 데이터가 가장 안정적")
print("- Level 2 호가는 무료 티어에서 제공 안 됨")
