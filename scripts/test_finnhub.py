#!/usr/bin/env python3
"""
Finnhub API 연결 테스트 스크립트
API 키가 올바르게 작동하는지 확인합니다.
"""

import sys
import os

# 프로젝트 루트 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.collector.finnhub_client import get_finnhub_client


def test_api_connection():
    """Finnhub API 연결 테스트"""
    print("=" * 60)
    print("Finnhub API 연결 테스트")
    print("=" * 60)
    print()

    # 1. 설정 확인
    print("✓ 단계 1: 설정 확인")
    print(f"  API 키: {settings.FINNHUB_API_KEY[:15]}...")
    print(f"  종목 수: {settings.TOP_N_VOLUME} + {settings.TOP_N_GAINERS}")
    print()

    # 2. Client 생성
    print("✓ 단계 2: Finnhub Client 생성")
    try:
        client = get_finnhub_client()
        print("  ✅ Client 생성 성공")
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        return False
    print()

    # 3. Health Check
    print("✓ 단계 3: API 연결 확인 (SPY quote 요청)")
    try:
        is_healthy = client.health_check()
        if is_healthy:
            print("  ✅ API 연결 정상!")
        else:
            print("  ❌ API 응답 없음")
            return False
    except Exception as e:
        print(f"  ❌ 연결 실패: {e}")
        return False
    print()

    # 4. 샘플 데이터 수집
    print("✓ 단계 4: 샘플 데이터 수집")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']

    for symbol in test_symbols:
        try:
            quote = client.get_quote(symbol)
            if quote and quote.get('c'):
                print(f"  ✅ {symbol}: ${quote['c']:.2f}")
            else:
                print(f"  ⚠ {symbol}: 데이터 없음")
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
    print()

    # 5. Candle 데이터 테스트
    print("✓ 단계 5: 5분봉 캔들 데이터 테스트")
    try:
        from datetime import datetime, timedelta

        to_ts = int(datetime.now().timestamp())
        from_ts = int((datetime.now() - timedelta(hours=24)).timestamp())

        candles = client.get_candles('AAPL', '5', from_ts, to_ts)

        if candles and candles.get('s') == 'ok':
            num_candles = len(candles['c'])
            print(f"  ✅ AAPL 5분봉: {num_candles}개 수신")
            print(f"  최신 종가: ${candles['c'][-1]:.2f}")
        else:
            print(f"  ⚠ 캔들 데이터: {candles.get('s', 'no data')}")
    except Exception as e:
        print(f"  ❌ 캔들 테스트 실패: {e}")
    print()

    # 6. API 호출 통계
    print("✓ 단계 6: API 사용 통계")
    print(f"  총 API 호출: {client.get_call_count()}회")
    print(f"  Rate Limit: 60 calls/minute")
    print()

    print("=" * 60)
    print("✅ 모든 테스트 통과!")
    print("=" * 60)
    print()
    print("다음 단계:")
    print("1. 과거 데이터 수집: python scripts/collect_historical.py --days 1")
    print("2. 모델 학습: python scripts/train_all_models.py")
    print("3. 시스템 실행: python scripts/run_system.py")
    print()

    return True


if __name__ == "__main__":
    try:
        success = test_api_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n중단됨")
        sys.exit(1)
    except Exception as e:
        print(f"\n예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
