# 📚 FiveForFree 문서 가이드

이 폴더는 프로젝트의 작업 관리 및 분석 문서를 포함합니다.

---

## 📁 문서 구조

```
Docs/
├── README.md                    # 이 파일 (문서 가이드)
├── TODO.md                      # 현재 작업 목록 (빠른 참조용)
├── ROADMAP.md                   # 프로젝트 로드맵 (장기 계획)
├── MODEL_PERFORMANCE_ANALYSIS.md # 모델 성능 분석 보고서
└── History/                     # 완료된 작업 아카이브
    ├── 2024-12-15_target_percent_change.md
    └── 2025-12-18_todo.md
```

---

## 🔍 문서별 용도

| 문서 | 용도 | 업데이트 주기 |
|------|------|---------------|
| **TODO.md** | 이번 주 작업, 빠른 참조 | 매일 |
| **ROADMAP.md** | 전체 프로젝트 로드맵, 마일스톤 | 주간/월간 |
| **MODEL_PERFORMANCE_ANALYSIS.md** | 모델 성능 분석 결과 | 분석 시 |
| **History/** | 완료된 작업 기록 보관 | 완료 시 이동 |

---

## 📖 기술 문서 위치

기술적인 상세 문서는 `docs/` 폴더(소문자)에 있습니다:

| 문서 | 내용 |
|------|------|
| `docs/PROJECT_ARCHITECTURE.md` | 시스템 아키텍처 |
| `docs/DATA_COLLECTION.md` | 데이터 수집 가이드 |
| `docs/FEATURES_REFERENCE.md` | 피처 레퍼런스 |
| `docs/HYBRID_ENSEMBLE_ARCHITECTURE.md` | 앙상블 모델 구조 |
| `docs/TESTING.md` | 테스트 가이드 |

---

## ✏️ 작업 기록 가이드라인

### 새 작업 시작 시
1. `TODO.md`에 작업 추가
2. 큰 마일스톤은 `ROADMAP.md`에도 추가

### 작업 완료 시
1. `TODO.md`에서 체크 (✅)
2. 주간 단위로 완료 항목 정리
3. 필요시 `History/` 폴더에 아카이브

### 파일 명명 규칙
- 날짜 기반: `YYYY-MM-DD_description.md`
- 예: `2025-12-24_backtest_results.md`

---

## 🔗 관련 링크

- [메인 README](/README.md)
- [프로젝트 명세](/PROJECT_SPEC.md)
- [기여 가이드](/CONTRIBUTING.md)
- [API 예제](/API_EXAMPLES.md)

---

**Last Updated**: 2025-12-24
