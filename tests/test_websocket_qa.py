"""WebSocket QA 검증 테스트"""

import asyncio
import json
import sys
from datetime import datetime

import websockets


class WebSocketTester:
    """WebSocket 연결 테스트 도구"""

    def __init__(self, url: str = "ws://localhost:8000/ws"):
        self.url = url
        self.results = []

    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """테스트 결과 기록"""
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        print(f"[{status}] {test_name}: {message}")

    async def test_connection(self):
        """TC-WS-001: 연결 테스트"""
        try:
            async with websockets.connect(self.url, timeout=10) as ws:
                # 연결 성공 메시지 수신 대기
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)

                if data.get("type") == "connected":
                    self.log_result(
                        "TC-WS-001: 연결 테스트",
                        True,
                        f"연결 성공 - {data.get('message')}"
                    )
                    return True
                else:
                    self.log_result(
                        "TC-WS-001: 연결 테스트",
                        False,
                        f"예상치 못한 메시지: {data}"
                    )
                    return False
        except Exception as e:
            self.log_result("TC-WS-001: 연결 테스트", False, str(e))
            return False

    async def test_ping_pong(self):
        """TC-WS-002: Ping/Pong 헬스 체크"""
        try:
            async with websockets.connect(self.url, timeout=10) as ws:
                # 연결 메시지 스킵
                await ws.recv()

                # Ping 전송
                await ws.send(json.dumps({"type": "ping"}))

                # Pong 수신 대기
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)

                if data.get("type") == "pong":
                    self.log_result(
                        "TC-WS-002: Ping/Pong 헬스 체크",
                        True,
                        f"Pong 수신 - timestamp: {data.get('timestamp')}"
                    )
                    return True
                else:
                    self.log_result(
                        "TC-WS-002: Ping/Pong 헬스 체크",
                        False,
                        f"예상치 못한 응답: {data}"
                    )
                    return False
        except Exception as e:
            self.log_result("TC-WS-002: Ping/Pong 헬스 체크", False, str(e))
            return False

    async def test_subscribe(self):
        """TC-WS-003: 티커 구독 테스트"""
        try:
            async with websockets.connect(self.url, timeout=10) as ws:
                # 연결 메시지 스킵
                await ws.recv()

                # 구독 요청
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "tickers": ["AAPL", "NVDA"]
                }))

                # 구독 확인 수신 대기
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)

                if data.get("type") == "subscribed":
                    tickers = data.get("tickers", [])
                    if "AAPL" in tickers and "NVDA" in tickers:
                        self.log_result(
                            "TC-WS-003: 티커 구독 테스트",
                            True,
                            f"구독 성공 - {tickers}"
                        )
                        return True
                    else:
                        self.log_result(
                            "TC-WS-003: 티커 구독 테스트",
                            False,
                            f"티커 누락: {tickers}"
                        )
                        return False
                else:
                    self.log_result(
                        "TC-WS-003: 티커 구독 테스트",
                        False,
                        f"예상치 못한 응답: {data}"
                    )
                    return False
        except Exception as e:
            self.log_result("TC-WS-003: 티커 구독 테스트", False, str(e))
            return False

    async def test_unsubscribe(self):
        """TC-WS-004: 티커 구독 해제 테스트"""
        try:
            async with websockets.connect(self.url, timeout=10) as ws:
                # 연결 메시지 스킵
                await ws.recv()

                # 먼저 구독
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "tickers": ["MSFT"]
                }))
                await ws.recv()  # 구독 확인 스킵

                # 구독 해제
                await ws.send(json.dumps({
                    "type": "unsubscribe",
                    "tickers": ["MSFT"]
                }))

                # 구독 해제 확인 수신
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)

                if data.get("type") == "unsubscribed":
                    self.log_result(
                        "TC-WS-004: 티커 구독 해제 테스트",
                        True,
                        f"구독 해제 성공 - {data.get('tickers')}"
                    )
                    return True
                else:
                    self.log_result(
                        "TC-WS-004: 티커 구독 해제 테스트",
                        False,
                        f"예상치 못한 응답: {data}"
                    )
                    return False
        except Exception as e:
            self.log_result("TC-WS-004: 티커 구독 해제 테스트", False, str(e))
            return False

    async def test_heartbeat(self):
        """TC-WS-005: 하트비트 수신 테스트"""
        try:
            async with websockets.connect(self.url, timeout=40) as ws:
                # 연결 메시지 스킵
                await ws.recv()

                # 하트비트 대기 (최대 35초)
                start_time = datetime.now()
                heartbeat_received = False

                while (datetime.now() - start_time).seconds < 35:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        data = json.loads(msg)
                        if data.get("type") == "heartbeat":
                            heartbeat_received = True
                            self.log_result(
                                "TC-WS-005: 하트비트 수신 테스트",
                                True,
                                f"하트비트 수신 - 연결 수: {data.get('connections')}"
                            )
                            return True
                    except asyncio.TimeoutError:
                        continue

                self.log_result(
                    "TC-WS-005: 하트비트 수신 테스트",
                    False,
                    "35초 내 하트비트 미수신"
                )
                return False
        except Exception as e:
            self.log_result("TC-WS-005: 하트비트 수신 테스트", False, str(e))
            return False

    async def test_price_update(self):
        """TC-WS-006: 실시간 가격 업데이트 테스트"""
        try:
            async with websockets.connect(self.url, timeout=25) as ws:
                # 연결 메시지 스킵
                await ws.recv()

                # 가격 업데이트 대기 (최대 20초)
                start_time = datetime.now()

                while (datetime.now() - start_time).seconds < 20:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        data = json.loads(msg)
                        if data.get("type") == "price_update":
                            prices = data.get("prices", [])
                            if prices:
                                sample = prices[0] if prices else {}
                                self.log_result(
                                    "TC-WS-006: 실시간 가격 업데이트 테스트",
                                    True,
                                    f"{len(prices)}개 티커 가격 수신 (예: {sample})"
                                )
                                return True
                    except asyncio.TimeoutError:
                        continue

                self.log_result(
                    "TC-WS-006: 실시간 가격 업데이트 테스트",
                    False,
                    "20초 내 가격 업데이트 미수신"
                )
                return False
        except Exception as e:
            self.log_result("TC-WS-006: 실시간 가격 업데이트 테스트", False, str(e))
            return False

    async def test_invalid_message(self):
        """TC-WS-007: 잘못된 메시지 처리 테스트"""
        try:
            async with websockets.connect(self.url, timeout=10) as ws:
                # 연결 메시지 스킵
                await ws.recv()

                # 잘못된 메시지 전송
                await ws.send("not a json")

                # 에러 응답 수신 대기
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)

                if data.get("type") == "error":
                    self.log_result(
                        "TC-WS-007: 잘못된 메시지 처리 테스트",
                        True,
                        f"에러 응답 수신: {data.get('message')}"
                    )
                    return True
                else:
                    self.log_result(
                        "TC-WS-007: 잘못된 메시지 처리 테스트",
                        False,
                        f"예상치 못한 응답: {data}"
                    )
                    return False
        except Exception as e:
            self.log_result("TC-WS-007: 잘못된 메시지 처리 테스트", False, str(e))
            return False

    async def test_unknown_message_type(self):
        """TC-WS-008: 알 수 없는 메시지 타입 처리"""
        try:
            async with websockets.connect(self.url, timeout=10) as ws:
                # 연결 메시지 스킵
                await ws.recv()

                # 알 수 없는 타입 전송
                await ws.send(json.dumps({"type": "unknown_type"}))

                # 에러 응답 수신 대기
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)

                if data.get("type") == "error" and "unknown_type" in data.get("message", "").lower():
                    self.log_result(
                        "TC-WS-008: 알 수 없는 메시지 타입 처리",
                        True,
                        f"에러 응답 수신: {data.get('message')}"
                    )
                    return True
                else:
                    self.log_result(
                        "TC-WS-008: 알 수 없는 메시지 타입 처리",
                        False,
                        f"예상치 못한 응답: {data}"
                    )
                    return False
        except Exception as e:
            self.log_result("TC-WS-008: 알 수 없는 메시지 타입 처리", False, str(e))
            return False

    async def run_all_tests(self, include_long_tests: bool = True):
        """모든 테스트 실행"""
        print("\n" + "=" * 60)
        print("WebSocket QA 검증 테스트 시작")
        print("=" * 60 + "\n")

        # 빠른 테스트
        await self.test_connection()
        await self.test_ping_pong()
        await self.test_subscribe()
        await self.test_unsubscribe()
        await self.test_invalid_message()
        await self.test_unknown_message_type()

        # 시간이 걸리는 테스트
        if include_long_tests:
            await self.test_price_update()
            await self.test_heartbeat()

        # 결과 요약
        print("\n" + "=" * 60)
        print("테스트 결과 요약")
        print("=" * 60)

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")

        if passed < total:
            print("\n실패한 테스트:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['test']}: {r['message']}")

        return passed, total


async def main():
    """메인 함수"""
    tester = WebSocketTester()

    # --quick 옵션으로 빠른 테스트만 실행
    include_long = "--quick" not in sys.argv

    passed, total = await tester.run_all_tests(include_long_tests=include_long)

    # 결과 파일 저장
    with open("websocket_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "passed": passed,
                "total": total,
                "pass_rate": f"{passed/total*100:.1f}%"
            },
            "results": tester.results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n결과가 websocket_test_results.json에 저장되었습니다.")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
