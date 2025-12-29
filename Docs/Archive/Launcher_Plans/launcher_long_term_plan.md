# FiveForFree 런처 장기 개선 방안 상세 기획

**작성자**: 분석팀장
**작성일**: 2025-12-27
**문서 버전**: 1.0

---

## 목차
1. GUI 프레임워크 전환 검토
2. 설정 영구 저장 방안
3. 구현 로드맵

---

## 1. GUI 프레임워크 전환 검토

### 1.1 현재 상태
- **사용 프레임워크**: Tkinter + ttk
- **문제점**:
  - 기본 위젯 디자인이 구식
  - 반응형 레이아웃 구현 복잡
  - 현대적인 UI 트렌드 반영 어려움
  - 애니메이션/전환 효과 제한적

### 1.2 후보 프레임워크 비교

#### A. CustomTkinter

| 항목 | 평가 | 설명 |
|------|------|------|
| **라이선스** | MIT | 상업용 가능 |
| **설치** | pip install customtkinter | 간단 |
| **문서화** | 양호 | 공식 문서 + 예제 |
| **커뮤니티** | 성장중 | GitHub 10K+ stars |
| **학습 곡선** | 낮음 | Tkinter 경험 그대로 활용 |

**장점**:
- Tkinter 기반이라 기존 코드 80-90% 재사용 가능
- 다크/라이트 테마 내장
- 현대적인 둥근 모서리 위젯
- DPI 자동 스케일링
- 애니메이션 효과 지원

**단점**:
- 상대적으로 신규 라이브러리 (2021~)
- 일부 고급 위젯 부재
- 복잡한 테이블/트리 위젯 제한적

**마이그레이션 예시**:
```python
# 기존 Tkinter
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
button = ttk.Button(root, text="Click")

# CustomTkinter
import customtkinter as ctk

root = ctk.CTk()
button = ctk.CTkButton(root, text="Click")
```

#### B. PyQt6

| 항목 | 평가 | 설명 |
|------|------|------|
| **라이선스** | GPL/상용 | 상업용은 별도 라이선스 필요 |
| **설치** | pip install PyQt6 | 용량 큼 (~100MB) |
| **문서화** | 우수 | 광범위한 문서 |
| **커뮤니티** | 대규모 | 20년+ 역사 |
| **학습 곡선** | 중간 | 새로운 패러다임 학습 필요 |

**장점**:
- 성숙한 라이브러리 (Qt 기반)
- 강력한 위젯 세트 (200+ 위젯)
- 시그널/슬롯 패턴으로 이벤트 처리 편리
- Qt Designer로 시각적 UI 디자인 가능
- 크로스 플랫폼 일관성 우수

**단점**:
- GPL 라이선스 (상용 배포 시 주의)
- 설치 용량 크고 배포 파일 증가
- 기존 Tkinter 코드 전면 재작성 필요
- 학습 시간 필요

**마이그레이션 예시**:
```python
# 기존 Tkinter
import tkinter as tk

root = tk.Tk()
root.title("App")
button = tk.Button(root, text="Click", command=func)
button.pack()
root.mainloop()

# PyQt6
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

app = QApplication([])
window = QMainWindow()
window.setWindowTitle("App")
button = QPushButton("Click", window)
button.clicked.connect(func)
window.show()
app.exec()
```

#### C. PySide6 (Qt for Python)

| 항목 | 평가 | 설명 |
|------|------|------|
| **라이선스** | LGPL | 상업용 가능 (동적 링킹) |
| **설치** | pip install PySide6 | 용량 큼 (~100MB) |
| **문서화** | 우수 | Qt 공식 문서 활용 |
| **커뮤니티** | 대규모 | Qt 공식 Python 바인딩 |
| **학습 곡선** | 중간 | PyQt6와 거의 동일 |

**장점**:
- PyQt6와 거의 동일한 API
- LGPL 라이선스로 상업용 부담 적음
- Qt Company 공식 지원
- Qt Creator 연동 가능

**단점**:
- PyQt6와 동일한 마이그레이션 비용
- 일부 API 차이 존재

### 1.3 마이그레이션 난이도 분석

| 항목 | CustomTkinter | PyQt6/PySide6 |
|------|---------------|---------------|
| **코드 재사용률** | 80-90% | 0-10% |
| **예상 작업 시간** | 1-2일 | 1-2주 |
| **위험도** | 낮음 | 중간 |
| **테스트 범위** | 기존 테스트 유지 | 전면 재테스트 |
| **의존성 변화** | 최소 | 대규모 |

### 1.4 권장 프레임워크: CustomTkinter

**선정 이유**:

1. **최소 마이그레이션 비용**
   - 기존 Tkinter 코드의 80-90% 재사용
   - `tk.` → `ctk.`, `ttk.` → `ctk.` 패턴 변환
   - 이벤트 핸들링 로직 그대로 유지

2. **충분한 디자인 개선 효과**
   - 현대적인 둥근 모서리 위젯
   - 다크/라이트 테마 내장
   - DPI 자동 스케일링

3. **프로젝트 규모에 적합**
   - 런처 GUI는 상대적으로 단순
   - 복잡한 테이블/차트 없음 (별도 창으로 분리)

4. **배포 용량 최소화**
   - CustomTkinter: ~5MB
   - PyQt6: ~100MB

**대안 검토가 필요한 경우**:
- 복잡한 차트 위젯 필요 시 → PyQt6 + PyQtGraph
- 상용 배포 시 라이선스 우려 → PySide6

---

## 2. 설정 영구 저장 방안

### 2.1 저장해야 할 설정 항목 목록

#### A. 창 상태
| 항목 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| window_x | int | 100 | 창 X 좌표 |
| window_y | int | 100 | 창 Y 좌표 |
| window_width | int | 900 | 창 너비 |
| window_height | int | 850 | 창 높이 |
| window_maximized | bool | false | 최대화 상태 |
| last_tab | str | "trading" | 마지막 활성 탭 |

#### B. 프로세스 설정
| 항목 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| auto_restart_live | bool | false | 실시간 트레이딩 자동재시작 |
| auto_restart_paper | bool | false | 모의 트레이딩 자동재시작 |
| auto_restart_api | bool | false | API 서버 자동재시작 |
| auto_restart_frontend | bool | false | 프론트엔드 자동재시작 |
| startup_processes | list | [] | 시작 시 자동 실행할 프로세스 |

#### C. UI 설정
| 항목 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| theme | str | "system" | 테마 (light/dark/system) |
| font_size | int | 9 | 기본 폰트 크기 |
| log_max_lines | int | 1000 | 로그 최대 줄 수 |
| auto_scroll_log | bool | true | 로그 자동 스크롤 |
| confirm_exit | bool | true | 종료 시 확인 |

#### D. 경로 설정
| 항목 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| results_dir | str | "./results" | 결과 저장 폴더 |
| logs_dir | str | "./logs" | 로그 저장 폴더 |
| backup_dir | str | "./backups" | 백업 저장 폴더 |

### 2.2 저장 방식 비교

#### A. JSON (권장)

```json
{
  "version": "1.0",
  "window": {
    "x": 100,
    "y": 100,
    "width": 900,
    "height": 850,
    "maximized": false
  },
  "processes": {
    "auto_restart": {
      "live": false,
      "paper": false,
      "api": true,
      "frontend": true
    },
    "startup": ["api", "frontend"]
  },
  "ui": {
    "theme": "dark",
    "font_size": 10,
    "log_max_lines": 1000
  }
}
```

**장점**:
- 사람이 읽고 수정 가능
- Python 표준 라이브러리 지원 (json 모듈)
- 가볍고 빠름
- 중첩 구조 지원

**단점**:
- 주석 불가
- 날짜/시간 직렬화 추가 처리 필요

#### B. YAML

```yaml
version: "1.0"
# 창 설정
window:
  x: 100
  y: 100
  width: 900
  height: 850
  maximized: false

# 프로세스 자동재시작
processes:
  auto_restart:
    live: false
    paper: false
    api: true
    frontend: true
  startup:
    - api
    - frontend
```

**장점**:
- 주석 지원
- 더 읽기 쉬운 형식
- 기존 config.yaml과 통일성

**단점**:
- PyYAML 의존성 필요
- 파싱 성능 JSON보다 느림

#### C. SQLite

**장점**:
- 대량 데이터 효율적
- 쿼리/검색 기능
- 트랜잭션 지원

**단점**:
- 단순 설정에 과함
- 직접 편집 불편
- 의존성 추가

### 2.3 권장 저장 방식: JSON

**선정 이유**:
1. 외부 의존성 없음 (Python 표준)
2. 설정 규모에 적합 (수십 개 항목)
3. 디버깅/수동 편집 용이
4. 모든 필요 기능 충족

### 2.4 구현 방안 상세

#### 파일 위치
```
PROJECT_ROOT/
├── launcher.py
├── launcher_settings.json  # 런처 설정 파일
└── config.yaml             # 기존 앱 설정
```

#### 설정 관리 클래스
```python
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class WindowSettings:
    x: int = 100
    y: int = 100
    width: int = 900
    height: int = 850
    maximized: bool = False
    last_tab: str = "trading"

@dataclass
class ProcessSettings:
    auto_restart_live: bool = False
    auto_restart_paper: bool = False
    auto_restart_api: bool = False
    auto_restart_frontend: bool = False
    startup_processes: List[str] = None

    def __post_init__(self):
        if self.startup_processes is None:
            self.startup_processes = []

@dataclass
class UISettings:
    theme: str = "system"
    font_size: int = 9
    log_max_lines: int = 1000
    auto_scroll_log: bool = True
    confirm_exit: bool = True

class LauncherSettings:
    SETTINGS_FILE = "launcher_settings.json"
    VERSION = "1.0"

    def __init__(self, project_root: Path):
        self.file_path = project_root / self.SETTINGS_FILE
        self.window = WindowSettings()
        self.processes = ProcessSettings()
        self.ui = UISettings()
        self.load()

    def load(self):
        """설정 파일 로드"""
        if not self.file_path.exists():
            return

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'window' in data:
                self.window = WindowSettings(**data['window'])
            if 'processes' in data:
                self.processes = ProcessSettings(**data['processes'])
            if 'ui' in data:
                self.ui = UISettings(**data['ui'])

        except (json.JSONDecodeError, TypeError) as e:
            print(f"설정 로드 실패: {e}")

    def save(self):
        """설정 파일 저장"""
        data = {
            'version': self.VERSION,
            'window': asdict(self.window),
            'processes': asdict(self.processes),
            'ui': asdict(self.ui)
        }

        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"설정 저장 실패: {e}")
```

#### 런처에 통합
```python
class TradingLauncher(tk.Tk):
    def __init__(self):
        super().__init__()

        # 설정 로드
        self.settings = LauncherSettings(PROJECT_ROOT)

        # 창 위치/크기 복원
        self._restore_window_state()

        # ... 기존 초기화 코드 ...

        # 종료 시 저장
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _restore_window_state(self):
        """저장된 창 상태 복원"""
        w = self.settings.window
        self.geometry(f"{w.width}x{w.height}+{w.x}+{w.y}")
        if w.maximized:
            self.state('zoomed')

    def _save_window_state(self):
        """현재 창 상태 저장"""
        self.settings.window.x = self.winfo_x()
        self.settings.window.y = self.winfo_y()
        self.settings.window.width = self.winfo_width()
        self.settings.window.height = self.winfo_height()
        self.settings.window.maximized = (self.state() == 'zoomed')

    def _on_close(self):
        """종료 처리"""
        self._save_window_state()
        self.settings.save()

        # 기존 종료 로직...
        self.destroy()
```

---

## 3. 구현 로드맵

### Phase 1: 설정 영구 저장 (1일)
1. `LauncherSettings` 클래스 구현
2. 창 위치/크기 저장 및 복원
3. 자동재시작 설정 저장
4. 테스트 및 검증

### Phase 2: CustomTkinter 마이그레이션 (2일)
1. customtkinter 의존성 추가
2. 메인 창 및 기본 위젯 전환
3. 버튼/라벨/프레임 전환
4. 다이얼로그 전환
5. 테마 설정 UI 추가
6. 전체 테스트

### Phase 3: 추가 UI 개선 (1일)
1. 테마 전환 기능
2. 폰트 크기 조절
3. 시작 프로세스 설정
4. 사용자 매뉴얼 업데이트

---

## 4. 결론

### 권장 사항 요약

| 항목 | 권장안 | 이유 |
|------|--------|------|
| GUI 프레임워크 | CustomTkinter | 최소 마이그레이션 비용, 충분한 UI 개선 |
| 설정 저장 형식 | JSON | 표준 라이브러리, 가벼움, 편집 용이 |
| 구현 순서 | 설정 저장 → 프레임워크 전환 | 독립적으로 테스트 가능 |

### 예상 효과
- 현대적인 UI로 사용자 경험 향상
- 사용자 설정 유지로 편의성 증대
- 유지보수성 향상

---

*문서 끝*
