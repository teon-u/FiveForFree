# 현재 작업 목록 (TODO)

> **Last Updated**: 2025-12-24
>
> 상세 로드맵: [ROADMAP.md](./ROADMAP.md)

---

## 🔥 이번 주 목표

### 백테스트 및 성과 측정
- [ ] `model_performance` 테이블 데이터 채우기
  - 스크립트: `scripts/populate_model_performance.py`
- [ ] Precision/Signal Rate 실제 값 계산
- [ ] UI에 실제 백테스트 결과 표시

### 프론트엔드 개선
- [ ] 다크/라이트 테마 지원
- [ ] 대시보드 상단 요약 패널 추가
- [ ] 실시간 업데이트 타임스탬프 표시

---

## 📋 대기 중 (Backlog)

### High Priority
| 작업 | 관련 파일 | 비고 |
|------|----------|------|
| 브라우저 알림 기능 | frontend/src/ | Web Notification API |
| 관심 종목(Watchlist) | DB 스키마 추가 필요 | localStorage or DB |

### Medium Priority
| 작업 | 관련 파일 | 비고 |
|------|----------|------|
| 고급 필터링 (확률/섹터) | frontend/src/components/ | 기존 필터 확장 |
| 스파크라인 미니 차트 | TickerCard 컴포넌트 | Recharts 사용 |
| 모바일 터치 제스처 | - | react-swipeable |

### Low Priority
| 작업 | 관련 파일 | 비고 |
|------|----------|------|
| PWA 지원 | vite.config.ts | vite-plugin-pwa |
| SHAP 모델 해석력 | src/models/ | 예측 이유 설명 |

---

## ✅ 최근 완료 (Last 7 days)

| 작업 | 완료일 | PR/Commit |
|------|--------|-----------|
| Walk-Forward validation | 2025-12-24 | #19 |
| 동적 종목 발견 시스템 | 2025-12-24 | faf8b12 |
| 차트 및 내보내기 기능 | 2025-12-24 | 937bd86 |
| 등급 기반 스타일링 | 2025-12-24 | f1b1586 |
| 필터링/정렬 시스템 | 2025-12-24 | 31eeca9 |
| Vitest 테스트 인프라 | 2025-12-24 | 11877b6 |
| WebSocket 구독 관리 | 2025-12-24 | 07200fa |
| ESLint 오류 수정 | 2025-12-24 | 833f420 |

---

## 🐛 알려진 이슈

| 이슈 | 심각도 | 상태 |
|------|--------|------|
| - | - | - |

---

## 📝 메모

- 백테스트 실행 전 최소 60일 데이터 필요
- Finnhub API 제한: 60 calls/min
- 모델 학습 시 GPU 메모리 관리 주의

---

**Quick Commands:**
```bash
# 데이터 상태 확인
python scripts/check_data.py

# 모델 학습
python scripts/train_all_models.py

# 시스템 실행
python scripts/run_system.py

# 프론트엔드 개발
cd frontend && npm run dev
```
