# 현재 작업 목록 (TODO)

> **Last Updated**: 2025-12-24
> **목표**: 12/25 시스템 정상 동작 확인 → 12/26 실전 투자 테스트

---

## 🎯 D-Day 목표 (12/25 크리스마스)

### 시스템 정상 동작 점검
- [ ] API 서버 정상 기동 확인 (`/api/health`)
- [ ] 데이터 수집 파이프라인 동작 확인
- [ ] 예측 API 응답 정상 여부 (`/api/predictions`)
- [ ] WebSocket 실시간 업데이트 확인
- [ ] 프론트엔드 UI 정상 표시

### 모델 상태 점검
- [ ] 학습된 모델 수 확인 (목표: 300개 이상)
- [ ] 각 모델별 최근 예측 정확도 확인
- [ ] 모델 로딩 시간 및 예측 속도 측정

### 데이터 품질 점검
- [ ] 분봉 데이터 최신성 확인 (최근 60일 이상)
- [ ] 누락된 종목 없는지 확인
- [ ] Feature 계산 정상 여부

---

## 🚀 실전 투자 효용가치 점검 (12/26~)

### 예측 신뢰도 검증
| 항목 | 기준 | 확인 방법 |
|------|------|----------|
| 상승 예측 정확도 | 60% 이상 | 백테스트 결과 확인 |
| A등급 종목 Hit율 | 65% 이상 | 고확률 종목 추적 |
| 신호 발생 빈도 | 하루 10건 이상 | 실시간 모니터링 |

### 실전 테스트 시나리오
- [ ] **Paper Trading**: 예측 기반 가상 매매 기록
  - 진입 시점 기록 (확률 70% 이상 신호 시)
  - 1시간 후 결과 확인
  - 수익/손실 집계

- [ ] **성과 지표 수집**
  - Precision: 상승 예측 중 실제 상승 비율
  - Signal Rate: 전체 중 신호 발생 비율
  - Win Rate: 거래 중 수익 거래 비율

### 투자 판단 기준
| 결과 | 판단 | 액션 |
|------|------|------|
| Precision 65%↑, Win Rate 55%↑ | ✅ 실전 투자 가능 | 소액으로 시작 |
| Precision 55~65% | ⚠️ 추가 검증 필요 | 1주일 더 관찰 |
| Precision 55%↓ | ❌ 모델 개선 필요 | 피처/학습 개선 |

---

## ✅ Bias 수정 작업 (12/24 완료)

> 예측 정확도에 직접 영향을 미치는 핵심 수정

### 🔴 P0: Label Intrabar Bias 수정 ✅
- [x] `label_generator.py` - high/low 대신 close 사용
- [x] Entry price를 다음봉 open으로 변경
- **영향**: 성공률 10~15% 과대평가 해소

### 🔴 P0: 백테스트 Entry Price 수정 ✅
- [x] `simulator.py` - 진입가를 다음봉 시가로 변경
- [x] 슬리피지 0.05% 추가
- **영향**: 수익률 0.2~0.5% 과대평가 해소

### 🟠 P1: Cumulative Feature 일별 Reset ✅
- [x] `feature_engineer.py` - VWAP 일별 reset
- [x] OBV ratio로 변환 (절대값 → 변화율)
- [x] VPT rolling sum으로 변환
- **영향**: Sign-flip 피처 문제 해결

### 🟡 P2: 모델 학습 개선 ✅
- [x] Early Stopping 구현 (LSTM/Transformer)
- [ ] 확률 캘리브레이션 적용 (보류)

---

## 📋 사전 준비 체크리스트 (12/24 오늘)

### 필수 (Bias 수정 후)
- [ ] 시스템 전체 기동 테스트
- [ ] 최신 데이터 수집 (`python scripts/collect_historical.py`)
- [ ] 모델 재학습 (`python scripts/train_all_models.py`)
- [ ] `model_performance` 테이블 데이터 확인

### 선택
- [ ] 로그 모니터링 설정
- [ ] 알림 설정 (슬랙/텔레그램 등)

---

## ✅ 최근 완료

| 작업 | 완료일 |
|------|--------|
| Walk-Forward validation | 2025-12-24 |
| 필터링/정렬 시스템 | 2025-12-24 |
| 차트 및 내보내기 | 2025-12-24 |
| 문서 구조 정리 | 2025-12-24 |

---

## ⏸️ 보류 (기능 추가)

> 실전 테스트 이후 순차적으로 진행

- 다크/라이트 테마
- 대시보드 요약 패널
- 브라우저 알림
- 관심 종목(Watchlist)
- 모바일 최적화

---

## 📝 Quick Commands

```bash
# 시스템 상태 확인
python scripts/check_data.py

# 데이터 수집 (최근 30일)
python scripts/collect_historical.py 30

# 모델 학습
python scripts/train_all_models.py

# API 서버 실행
python scripts/run_system.py

# 프론트엔드 실행
cd frontend && npm run dev
```

---

**핵심 질문**: 12/26 장 시작 전, 예측을 믿고 실제 매매할 수 있는가?
