# FiveForFree 문서 가이드

프로젝트의 핵심 기술 문서와 작업 관리 문서를 포함합니다.

---

## 문서 구조

```
docs/
├── README.md                         # 이 파일 (문서 가이드)
│
├── [핵심 기술 문서]
│   ├── PROJECT_ARCHITECTURE.md       # 시스템 아키텍처
│   ├── DATA_COLLECTION.md            # 데이터 수집 가이드
│   ├── FEATURES_REFERENCE.md         # 57개 피처 명세
│   ├── HYBRID_ENSEMBLE_ARCHITECTURE.md # 앙상블 모델 구조
│   ├── API_REFERENCE.md              # REST/WebSocket API 명세
│   └── TESTING.md                    # 테스트 가이드
│
├── [프로젝트 관리]
│   ├── TODO.md                       # 단기 작업 목록
│   └── ROADMAP.md                    # 장기 로드맵
│
├── [진행 중인 기획]
│   ├── LAUNCHER_V1.3.0_PLAN.md       # 런처 v1.3.0 기획
│   ├── integrated_backtest_ui_plan.md # 통합 백테스트 UI
│   └── correlation_diversification_plan.md # 종목 다각화 전략
│
├── History/                          # 완료된 작업 아카이브
│
└── Archive/                          # 구버전/완료된 문서
    ├── QA_Reports/                   # 완료된 QA 테스트 리포트
    ├── Design_Completed/             # 구현 완료된 설계 문서
    ├── Analysis_Reports/             # 과거 분석 리포트
    ├── Future_Plans/                 # 미구현 기획 문서
    ├── Launcher_Plans/               # 런처 구버전 기획
    └── Signal_Improvement/           # 신호율 개선 관련
```

---

## 핵심 문서 (필수 참조)

| 문서 | 용도 | 업데이트 주기 |
|------|------|---------------|
| **PROJECT_ARCHITECTURE.md** | 전체 시스템 구조, 모듈, 데이터 흐름 | 구조 변경 시 |
| **DATA_COLLECTION.md** | 데이터 수집 전략, API 제한사항 | 수집 로직 변경 시 |
| **FEATURES_REFERENCE.md** | 57개 피처 목록, 계산 방식 | 피처 변경 시 |
| **HYBRID_ENSEMBLE_ARCHITECTURE.md** | 모델 앙상블 구조, 가중치 | 모델 변경 시 |
| **API_REFERENCE.md** | REST/WebSocket 엔드포인트 | API 변경 시 |
| **TESTING.md** | 테스트 실행 방법, CI/CD | 테스트 변경 시 |

---

## 프로젝트 관리 문서

| 문서 | 용도 | 업데이트 주기 |
|------|------|---------------|
| **TODO.md** | 이번 주 작업, 체크리스트 | 매일 |
| **ROADMAP.md** | 마일스톤, 장기 계획 | 주간/월간 |

---

## Archive 폴더 설명

| 폴더 | 내용 | 보관 이유 |
|------|------|----------|
| **QA_Reports/** | 완료된 기능 테스트 리포트 | 회귀 테스트 참고 |
| **Design_Completed/** | 구현 완료된 설계 문서 | 의사결정 근거 |
| **Analysis_Reports/** | 과거 모델/시스템 분석 | 성능 비교 참고 |
| **Future_Plans/** | 미구현된 기능 기획 | 향후 개발 참고 |
| **Launcher_Plans/** | 런처 구버전 기획 | 히스토리 보존 |
| **Signal_Improvement/** | 신호율 개선 관련 문서 | 최적화 참고 |

---

## 작업 가이드라인

### 새 작업 시작 시
1. `TODO.md`에 작업 추가
2. 큰 마일스톤은 `ROADMAP.md`에도 추가

### 작업 완료 시
1. `TODO.md`에서 체크 완료
2. 설계 문서는 `Archive/Design_Completed/`로 이동
3. 분석 리포트는 `Archive/Analysis_Reports/`로 이동

### 파일 명명 규칙
- 날짜 기반: `YYYY-MM-DD_description.md`
- 예: `2025-12-24_backtest_results.md`

---

## 관련 링크

- [메인 README](/README.md)
- [Claude 작업 지침](/CLAUDE.md)

---

**Last Updated**: 2025-12-27
