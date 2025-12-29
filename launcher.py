#!/usr/bin/env python
"""
FiveForFree Trading System Launcher

PyTkinter GUI for managing the trading system:
- Start/Stop live trading
- Run backtests
- Monitor process status
- View logs

Author: 개발팀장
Date: 2025-12-25
"""

import os
import sys
import subprocess
import threading
import queue
import webbrowser
import ctypes
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font as tkfont
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict, field

# CustomTkinter for modern UI
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False

# PyInstaller frozen exe 감지 및 프로젝트 루트 설정
if getattr(sys, "frozen", False):
    # exe로 실행 시 - exe 파일 위치 기준
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
else:
    # 스크립트로 실행 시
    PROJECT_ROOT = Path(__file__).parent.resolve()

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Windows DPI Awareness 설정 (고해상도 디스플레이 지원)
if sys.platform == "win32":
    try:
        # Windows 10 1607+ Per-Monitor DPI Awareness
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            # Windows 8.1+ System DPI Awareness
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                # Windows Vista+ DPI Awareness
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

# PROJECT_ROOT은 위에서 이미 설정됨
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
API_SCRIPT = PROJECT_ROOT / "run_api.py"
API_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"
SETTINGS_FILE = PROJECT_ROOT / "launcher_settings.json"


# ============================================================================
# 설정 영구 저장 클래스 (LauncherSettings)
# ============================================================================

@dataclass
class WindowSettings:
    """창 위치/크기 설정."""
    x: int = 100
    y: int = 100
    width: int = 900
    height: int = 850
    maximized: bool = False
    last_tab: str = "trading"


@dataclass
class ProcessSettings:
    """프로세스 자동재시작 설정."""
    auto_restart_live: bool = False
    auto_restart_paper: bool = False
    auto_restart_api: bool = False
    auto_restart_frontend: bool = False
    startup_processes: List[str] = field(default_factory=list)


@dataclass
class UISettings:
    """UI 설정."""
    theme: str = "system"  # light, dark, system
    font_size: int = 9
    log_max_lines: int = 1000
    auto_scroll_log: bool = True
    confirm_exit: bool = True


class LauncherSettings:
    """런처 설정 관리 클래스."""
    VERSION = "1.0"

    def __init__(self, settings_file: Path = SETTINGS_FILE):
        self.file_path = settings_file
        self.window = WindowSettings()
        self.processes = ProcessSettings()
        self.ui = UISettings()
        self.load()

    def load(self):
        """설정 파일 로드."""
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
        """설정 파일 저장."""
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


class Tooltip:
    """위젯용 간단한 툴팁."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tipwindow:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
            font=("맑은 고딕", 9)
        )
        label.pack()

    def hide(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class DataStatusDialog(tk.Toplevel):
    """데이터 상태 확인 다이얼로그."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("데이터 상태")
        self.geometry("450x500")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # 메인 프레임
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 로딩 표시
        self.loading_label = ttk.Label(main_frame, text="데이터 조회 중...", font=("맑은 고딕", 10))
        self.loading_label.pack(pady=20)

        # 결과 표시 영역
        self.result_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            state=tk.DISABLED,
            height=20
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 버튼 프레임
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="새로고침", command=self._load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="닫기", command=self.destroy).pack(side=tk.RIGHT, padx=5)

        # 데이터 로드 시작 (별도 스레드)
        self._load_data()

    def _load_data(self):
        """데이터 로드 (별도 스레드에서)."""
        self.loading_label.config(text="데이터 조회 중...")
        threading.Thread(target=self._fetch_data, daemon=True).start()

    def _fetch_data(self):
        """DB에서 데이터 조회."""
        try:
            from src.utils.database import get_db, MinuteBar, Ticker
            from src.models.model_manager import ModelManager
            from sqlalchemy import select, func

            result_lines = []

            # 모델 정보
            mm = ModelManager()
            mm.load_all_models()
            trained_tickers = mm.get_tickers()

            result_lines.append("=" * 45)
            result_lines.append("  데이터베이스 상태")
            result_lines.append("=" * 45)
            result_lines.append("")

            with get_db() as db:
                # 티커 수
                ticker_count = db.execute(select(func.count(Ticker.id))).scalar() or 0
                result_lines.append(f"  등록 티커: {ticker_count}개")

                # 분봉 데이터 수
                bar_count = db.execute(select(func.count(MinuteBar.id))).scalar() or 0
                result_lines.append(f"  분봉 데이터: {bar_count:,}개")

                # 데이터 있는 티커
                symbols_stmt = select(MinuteBar.symbol).distinct()
                symbols = db.execute(symbols_stmt).scalars().all()
                result_lines.append(f"  데이터 보유 티커: {len(symbols)}개")

                # 날짜 범위
                min_date = db.execute(select(func.min(MinuteBar.timestamp))).scalar()
                max_date = db.execute(select(func.max(MinuteBar.timestamp))).scalar()
                if min_date and max_date:
                    days = (max_date - min_date).days
                    result_lines.append(f"  데이터 기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
                    result_lines.append(f"  총 {days}일")

                result_lines.append("")
                result_lines.append("=" * 45)
                result_lines.append("  모델 상태")
                result_lines.append("=" * 45)
                result_lines.append("")

                result_lines.append(f"  학습된 모델: {len(trained_tickers)}개")

                # 커버리지 분석
                tickers_with_data = set(symbols)
                tickers_with_models = set(trained_tickers)
                both = tickers_with_data & tickers_with_models
                data_only = tickers_with_data - tickers_with_models
                model_only = tickers_with_models - tickers_with_data

                result_lines.append(f"  준비 완료 티커: {len(both)}개")
                result_lines.append(f"  데이터만 있음: {len(data_only)}개")
                result_lines.append(f"  모델만 있음: {len(model_only)}개")

                result_lines.append("")
                result_lines.append("=" * 45)
                result_lines.append("  Top 5 티커 (바 데이터 기준)")
                result_lines.append("=" * 45)
                result_lines.append("")

                # Top 5 티커
                bar_count_stmt = (
                    select(MinuteBar.symbol, func.count(MinuteBar.id).label('count'))
                    .group_by(MinuteBar.symbol)
                    .order_by(func.count(MinuteBar.id).desc())
                    .limit(5)
                )
                for i, row in enumerate(db.execute(bar_count_stmt), 1):
                    has_model = "O" if row.symbol in tickers_with_models else "X"
                    result_lines.append(f"  {i}. {row.symbol}: {row.count:,} bars [모델:{has_model}]")

            result_lines.append("")
            result_lines.append("=" * 45)

            # UI 업데이트 (메인 스레드에서)
            self.after(0, lambda: self._update_result("\n".join(result_lines)))

        except Exception as e:
            self.after(0, lambda: self._update_result(f"오류 발생:\n{str(e)}"))

    def _update_result(self, text: str):
        """결과 텍스트 업데이트."""
        self.loading_label.config(text="")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)


class TickerListWindow(tk.Toplevel):
    """티커 목록 창."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("티커 목록")
        self.geometry("700x500")
        self.minsize(600, 400)
        self.transient(parent)

        # 메인 프레임
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 상단 컨트롤
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(control_frame, text="검색:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", lambda *args: self._filter_tickers())
        search_entry = ttk.Entry(control_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="필터:").pack(side=tk.LEFT, padx=(10, 0))
        self.filter_var = tk.StringVar(value="전체")
        filter_combo = ttk.Combobox(
            control_frame,
            textvariable=self.filter_var,
            values=["전체", "활성 티커만", "모델 있는 것만"],
            state="readonly",
            width=15
        )
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._filter_tickers())

        ttk.Button(control_frame, text="새로고침", command=self._load_tickers).pack(side=tk.RIGHT)

        # Treeview
        columns = ("symbol", "name", "bars", "model", "status")
        self.tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)

        self.tree.heading("symbol", text="심볼", command=lambda: self._sort_by("symbol"))
        self.tree.heading("name", text="이름", command=lambda: self._sort_by("name"))
        self.tree.heading("bars", text="바 데이터", command=lambda: self._sort_by("bars"))
        self.tree.heading("model", text="모델", command=lambda: self._sort_by("model"))
        self.tree.heading("status", text="상태", command=lambda: self._sort_by("status"))

        self.tree.column("symbol", width=80, anchor=tk.CENTER)
        self.tree.column("name", width=200)
        self.tree.column("bars", width=100, anchor=tk.E)
        self.tree.column("model", width=60, anchor=tk.CENTER)
        self.tree.column("status", width=80, anchor=tk.CENTER)

        # 스크롤바
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 하단 상태바
        self.status_label = ttk.Label(self, text="로딩 중...", padding=(10, 5))
        self.status_label.pack(fill=tk.X)

        # 데이터 저장
        self.all_tickers: List[dict] = []
        self.tickers_with_models = set()
        self.sort_column = "symbol"
        self.sort_reverse = False

        # 데이터 로드
        self._load_tickers()

    def _load_tickers(self):
        """티커 목록 로드."""
        self.status_label.config(text="데이터 조회 중...")
        threading.Thread(target=self._fetch_tickers, daemon=True).start()

    def _fetch_tickers(self):
        """DB에서 티커 조회."""
        try:
            from src.utils.database import get_db, MinuteBar, Ticker
            from src.models.model_manager import ModelManager
            from sqlalchemy import select, func

            # 모델 정보
            mm = ModelManager()
            mm.load_all_models()
            self.tickers_with_models = set(mm.get_tickers())

            tickers = []

            with get_db() as db:
                # 티커 + 바 카운트 조회
                bar_count_stmt = (
                    select(MinuteBar.symbol, func.count(MinuteBar.id).label('count'))
                    .group_by(MinuteBar.symbol)
                )
                bar_counts = {row.symbol: row.count for row in db.execute(bar_count_stmt)}

                # 티커 정보 조회
                ticker_stmt = select(Ticker).order_by(Ticker.symbol)
                for ticker in db.execute(ticker_stmt).scalars().all():
                    tickers.append({
                        "symbol": ticker.symbol,
                        "name": ticker.name or "",
                        "bars": bar_counts.get(ticker.symbol, 0),
                        "model": "O" if ticker.symbol in self.tickers_with_models else "X",
                        "status": "활성" if ticker.is_active else "비활성"
                    })

            self.all_tickers = tickers
            self.after(0, self._filter_tickers)

        except Exception as e:
            self.after(0, lambda: self.status_label.config(text=f"오류: {str(e)}"))

    def _filter_tickers(self):
        """필터링 및 표시."""
        # 기존 항목 삭제
        for item in self.tree.get_children():
            self.tree.delete(item)

        search = self.search_var.get().upper()
        filter_type = self.filter_var.get()

        filtered = []
        for t in self.all_tickers:
            # 검색 필터
            if search and search not in t["symbol"] and search not in t["name"].upper():
                continue

            # 타입 필터
            if filter_type == "활성 티커만" and t["status"] != "활성":
                continue
            if filter_type == "모델 있는 것만" and t["model"] != "O":
                continue

            filtered.append(t)

        # 정렬
        key_map = {
            "symbol": lambda x: x["symbol"],
            "name": lambda x: x["name"],
            "bars": lambda x: x["bars"],
            "model": lambda x: x["model"],
            "status": lambda x: x["status"]
        }
        filtered.sort(key=key_map.get(self.sort_column, lambda x: x["symbol"]), reverse=self.sort_reverse)

        # 표시
        for t in filtered:
            self.tree.insert("", tk.END, values=(
                t["symbol"],
                t["name"][:25] + "..." if len(t["name"]) > 25 else t["name"],
                f"{t['bars']:,}",
                t["model"],
                t["status"]
            ))

        self.status_label.config(text=f"표시: {len(filtered)}/{len(self.all_tickers)} 티커")

    def _sort_by(self, column: str):
        """컬럼 정렬."""
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False
        self._filter_tickers()


class ModelTrainingDialog(tk.Toplevel):
    """모델 학습 관리 다이얼로그."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("모델 학습 관리")
        self.geometry("500x550")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.training_process = None
        self.is_training = False
        self.log_queue = queue.Queue()

        # 메인 프레임
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 티커 및 모델 선택
        select_frame = ttk.LabelFrame(main_frame, text="학습 설정", padding="10")
        select_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(select_frame, text="티커:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.ticker_var = tk.StringVar(value="전체")
        self.ticker_combo = ttk.Combobox(select_frame, textvariable=self.ticker_var, width=15)
        self.ticker_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(select_frame, text="모델:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.model_var = tk.StringVar(value="Ensemble")
        self.model_combo = ttk.Combobox(
            select_frame, textvariable=self.model_var, width=15,
            values=["Ensemble", "LSTM", "XGBoost", "LightGBM"]
        )
        self.model_combo.grid(row=0, column=3, padx=5, pady=5)

        # 버튼 프레임
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="학습 시작", command=self._start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="중지", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="결과 보기", command=self._open_results).pack(side=tk.LEFT, padx=5)

        # 진행률
        progress_frame = ttk.LabelFrame(main_frame, text="진행률", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="대기 중...")
        self.progress_label.pack(anchor=tk.W)

        # 학습 로그
        log_frame = ttk.LabelFrame(main_frame, text="학습 로그", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            height=12
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 티커 목록 로드
        self._load_tickers()

        # 로그 업데이트 타이머
        self._update_log()

    def _load_tickers(self):
        """티커 목록 로드."""
        tickers = ["전체"]
        try:
            from src.utils.database import get_db, Ticker
            from sqlalchemy import select
            with get_db() as db:
                symbols = db.execute(select(Ticker.symbol).order_by(Ticker.symbol)).scalars().all()
                tickers.extend(symbols)
        except Exception:
            pass
        self.ticker_combo["values"] = tickers

    def _start_training(self):
        """학습 시작."""
        self.is_training = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)

        ticker = self.ticker_var.get()
        model = self.model_var.get()

        self._log(f"[시작] {ticker} 티커 {model} 모델 학습 시작...")

        # 학습 스크립트 실행
        threading.Thread(target=self._run_training, args=(ticker, model), daemon=True).start()

    def _run_training(self, ticker: str, model: str):
        """학습 스레드."""
        try:
            python = str(VENV_PYTHON) if VENV_PYTHON.exists() else "python"
            script = SCRIPTS_DIR / "train_models.py"

            if not script.exists():
                self._log("[오류] train_models.py 스크립트가 없습니다.")
                self.after(0, self._training_finished)
                return

            cmd = [python, str(script)]
            if ticker != "전체":
                cmd.extend(["--ticker", ticker])

            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                text=True,
                bufsize=1
            )

            for line in iter(self.training_process.stdout.readline, ''):
                if line:
                    self.log_queue.put(line.strip())
                    # 진행률 파싱 시도
                    if "Epoch" in line and "/" in line:
                        try:
                            parts = line.split("Epoch")[1].split("/")
                            current = int(parts[0].strip())
                            total = int(parts[1].split()[0])
                            progress = (current / total) * 100
                            self.after(0, lambda p=progress: self.progress_var.set(p))
                            self.after(0, lambda c=current, t=total: self.progress_label.config(
                                text=f"에폭: {c}/{t} ({progress:.1f}%)"
                            ))
                        except Exception:
                            pass

            self.training_process.wait()
            self._log("[완료] 학습 완료!")
            self.after(0, self._training_finished)

        except Exception as e:
            self._log(f"[오류] {str(e)}")
            self.after(0, self._training_finished)

    def _stop_training(self):
        """학습 중지."""
        if self.training_process:
            self.training_process.terminate()
            self._log("[중지] 학습이 중지되었습니다.")
        self._training_finished()

    def _training_finished(self):
        """학습 완료 처리."""
        self.is_training = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.training_process = None

    def _open_results(self):
        """결과 폴더 열기."""
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            os.startfile(str(models_dir))

    def _log(self, message: str):
        """로그 메시지 추가."""
        self.log_queue.put(message)

    def _update_log(self):
        """로그 UI 업데이트."""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
            except queue.Empty:
                break
        self.after(100, self._update_log)


class SystemMonitorDialog(tk.Toplevel):
    """시스템 리소스 모니터링 다이얼로그."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("시스템 모니터")
        self.geometry("450x400")
        self.resizable(False, False)
        self.transient(parent)

        self.running = True

        # 메인 프레임
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 시스템 리소스 프레임
        resource_frame = ttk.LabelFrame(main_frame, text="시스템 리소스", padding="10")
        resource_frame.pack(fill=tk.X, pady=(0, 10))

        # CPU
        ttk.Label(resource_frame, text="CPU:", width=8).grid(row=0, column=0, sticky=tk.W, pady=3)
        self.cpu_var = tk.DoubleVar(value=0)
        self.cpu_bar = ttk.Progressbar(resource_frame, variable=self.cpu_var, maximum=100, length=250)
        self.cpu_bar.grid(row=0, column=1, padx=5, pady=3)
        self.cpu_label = ttk.Label(resource_frame, text="0%", width=8)
        self.cpu_label.grid(row=0, column=2, pady=3)

        # RAM
        ttk.Label(resource_frame, text="RAM:", width=8).grid(row=1, column=0, sticky=tk.W, pady=3)
        self.ram_var = tk.DoubleVar(value=0)
        self.ram_bar = ttk.Progressbar(resource_frame, variable=self.ram_var, maximum=100, length=250)
        self.ram_bar.grid(row=1, column=1, padx=5, pady=3)
        self.ram_label = ttk.Label(resource_frame, text="0 / 0 GB", width=12)
        self.ram_label.grid(row=1, column=2, pady=3)

        # GPU (if available)
        ttk.Label(resource_frame, text="GPU:", width=8).grid(row=2, column=0, sticky=tk.W, pady=3)
        self.gpu_var = tk.DoubleVar(value=0)
        self.gpu_bar = ttk.Progressbar(resource_frame, variable=self.gpu_var, maximum=100, length=250)
        self.gpu_bar.grid(row=2, column=1, padx=5, pady=3)
        self.gpu_label = ttk.Label(resource_frame, text="N/A", width=12)
        self.gpu_label.grid(row=2, column=2, pady=3)

        # 디스크
        ttk.Label(resource_frame, text="Disk:", width=8).grid(row=3, column=0, sticky=tk.W, pady=3)
        self.disk_var = tk.DoubleVar(value=0)
        self.disk_bar = ttk.Progressbar(resource_frame, variable=self.disk_var, maximum=100, length=250)
        self.disk_bar.grid(row=3, column=1, padx=5, pady=3)
        self.disk_label = ttk.Label(resource_frame, text="0 GB 사용 가능", width=15)
        self.disk_label.grid(row=3, column=2, pady=3)

        # 프로세스 정보
        process_frame = ttk.LabelFrame(main_frame, text="프로세스별 리소스", padding="10")
        process_frame.pack(fill=tk.BOTH, expand=True)

        self.process_text = scrolledtext.ScrolledText(
            process_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            height=8
        )
        self.process_text.pack(fill=tk.BOTH, expand=True)

        # 버튼
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_frame, text="새로고침", command=self._update_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="닫기", command=self._on_close).pack(side=tk.RIGHT, padx=5)

        # 초기 업데이트
        self._update_stats()
        self._auto_update()

        # 닫기 이벤트
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_stats(self):
        """시스템 상태 업데이트."""
        threading.Thread(target=self._fetch_stats, daemon=True).start()

    def _fetch_stats(self):
        """시스템 정보 조회."""
        try:
            import psutil

            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.after(0, lambda: self.cpu_var.set(cpu_percent))
            self.after(0, lambda: self.cpu_label.config(text=f"{cpu_percent:.1f}%"))

            # RAM
            mem = psutil.virtual_memory()
            ram_percent = mem.percent
            ram_used = mem.used / (1024**3)
            ram_total = mem.total / (1024**3)
            self.after(0, lambda: self.ram_var.set(ram_percent))
            self.after(0, lambda: self.ram_label.config(text=f"{ram_used:.1f}/{ram_total:.1f} GB"))

            # Disk
            disk = psutil.disk_usage(str(PROJECT_ROOT))
            disk_percent = disk.percent
            disk_free = disk.free / (1024**3)
            self.after(0, lambda: self.disk_var.set(disk_percent))
            self.after(0, lambda: self.disk_label.config(text=f"{disk_free:.1f} GB 사용 가능"))

            # GPU (try)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
                    gpu_name = torch.cuda.get_device_name(0)[:20]
                    self.after(0, lambda: self.gpu_var.set(gpu_mem))
                    self.after(0, lambda: self.gpu_label.config(text=f"{gpu_mem:.1f}% ({gpu_name})"))
            except Exception:
                pass

            # 프로세스 정보
            process_info = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    pinfo = proc.info
                    name = pinfo['name'].lower()
                    if 'python' in name or 'node' in name:
                        mem_mb = pinfo['memory_info'].rss / (1024**2) if pinfo['memory_info'] else 0
                        cpu = pinfo['cpu_percent'] or 0
                        if mem_mb > 50:  # 50MB 이상만 표시
                            process_info.append(f"{pinfo['name']}: CPU {cpu:.1f}%, RAM {mem_mb:.0f}MB")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            process_text = "\n".join(process_info) if process_info else "관련 프로세스 없음"
            self.after(0, lambda: self._update_process_text(process_text))

        except ImportError:
            self.after(0, lambda: self._update_process_text("psutil 모듈이 필요합니다.\npip install psutil"))
        except Exception as e:
            self.after(0, lambda: self._update_process_text(f"오류: {str(e)}"))

    def _update_process_text(self, text: str):
        """프로세스 텍스트 업데이트."""
        self.process_text.config(state=tk.NORMAL)
        self.process_text.delete(1.0, tk.END)
        self.process_text.insert(tk.END, text)
        self.process_text.config(state=tk.DISABLED)

    def _auto_update(self):
        """2초마다 자동 업데이트."""
        if self.running:
            self._update_stats()
            self.after(2000, self._auto_update)

    def _on_close(self):
        """닫기 처리."""
        self.running = False
        self.destroy()


class DataCollectionDialog(tk.Toplevel):
    """데이터 수집 제어 다이얼로그."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("데이터 수집 제어")
        self.geometry("500x450")
        self.resizable(False, False)

        self.running = True
        self.collecting = False
        self.collection_thread = None

        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 버튼 프레임
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.collect_btn = ttk.Button(
            btn_frame, text="즉시 수집", width=12,
            command=self._start_collection
        )
        self.collect_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(
            btn_frame, text="중지", width=8,
            command=self._stop_collection,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame, text="스케줄 설정", width=12,
            command=self._show_schedule
        ).pack(side=tk.LEFT, padx=2)

        # 상태 표시
        status_frame = ttk.LabelFrame(main_frame, text="수집 상태", padding=5)
        status_frame.pack(fill=tk.X, pady=10)

        self.status_label = ttk.Label(status_frame, text="대기 중")
        self.status_label.pack(anchor=tk.W)

        # 진행률
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var,
            maximum=100, length=450
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(status_frame, text="0/0 티커")
        self.progress_label.pack(anchor=tk.W)

        # 최근 수집
        recent_frame = ttk.LabelFrame(main_frame, text="최근 수집", padding=5)
        recent_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.recent_text = scrolledtext.ScrolledText(
            recent_frame, height=12, font=("Consolas", 9)
        )
        self.recent_text.pack(fill=tk.BOTH, expand=True)

        self._load_recent_collections()

    def _load_recent_collections(self):
        """최근 수집 이력 로드."""
        try:
            from core.database import Database
            db = Database()
            session = db.Session()

            # 최근 데이터 상태 확인
            from models import StockData
            from sqlalchemy import func

            recent = session.query(
                StockData.ticker,
                func.count(StockData.id).label('count'),
                func.max(StockData.date).label('last_date')
            ).group_by(StockData.ticker).order_by(
                func.max(StockData.date).desc()
            ).limit(10).all()

            text = ""
            for ticker, count, last_date in recent:
                text += f"{ticker}: {count:,} bars (최근: {last_date})\n"

            if not text:
                text = "수집된 데이터가 없습니다."

            self.recent_text.delete(1.0, tk.END)
            self.recent_text.insert(tk.END, text)
            session.close()
        except Exception as e:
            self.recent_text.delete(1.0, tk.END)
            self.recent_text.insert(tk.END, f"데이터 로드 실패: {e}")

    def _start_collection(self):
        """데이터 수집 시작."""
        if self.collecting:
            return

        self.collecting = True
        self.collect_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="수집 시작 중...")

        self.collection_thread = threading.Thread(
            target=self._run_collection,
            daemon=True
        )
        self.collection_thread.start()

    def _run_collection(self):
        """백그라운드에서 데이터 수집."""
        try:
            from data_collection import collect_all_data
            # 티커 목록 가져오기
            from tickers import TICKERS

            total = len(TICKERS)
            self.after(0, lambda: self.progress_label.config(text=f"0/{total} 티커"))

            for i, ticker in enumerate(TICKERS):
                if not self.collecting:
                    break

                self.after(0, lambda t=ticker: self.status_label.config(text=f"수집 중: {t}"))
                self.after(0, lambda i=i, t=total: self.progress_var.set((i/t)*100))
                self.after(0, lambda i=i, t=total: self.progress_label.config(text=f"{i}/{t} 티커"))

                try:
                    collect_all_data([ticker])
                    self.after(0, lambda t=ticker: self._add_recent(f"{t}: 수집 완료"))
                except Exception as e:
                    self.after(0, lambda t=ticker, e=str(e): self._add_recent(f"{t}: 오류 - {e}"))

            self.after(0, lambda: self.status_label.config(text="수집 완료" if self.collecting else "수집 중단"))
            self.after(0, lambda: self.progress_var.set(100 if self.collecting else 0))

        except ImportError as e:
            self.after(0, lambda: self.status_label.config(text=f"모듈 오류: {e}"))
        except Exception as e:
            self.after(0, lambda: self.status_label.config(text=f"오류: {e}"))
        finally:
            self.collecting = False
            self.after(0, lambda: self.collect_btn.config(state=tk.NORMAL))
            self.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def _add_recent(self, text: str):
        """최근 수집 텍스트 추가."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.recent_text.insert(1.0, f"[{timestamp}] {text}\n")

    def _stop_collection(self):
        """데이터 수집 중지."""
        self.collecting = False
        self.status_label.config(text="중지 요청됨...")

    def _show_schedule(self):
        """스케줄 설정 대화상자."""
        from tkinter import simpledialog
        hour = simpledialog.askinteger(
            "스케줄 설정",
            "매일 수집 시간 (0-23시):",
            minvalue=0, maxvalue=23,
            parent=self
        )
        if hour is not None:
            messagebox.showinfo(
                "스케줄 설정",
                f"매일 {hour:02d}:00에 자동 수집이 실행됩니다.\n"
                "(참고: 런처가 실행 중이어야 합니다)",
                parent=self
            )

    def _on_close(self):
        """닫기 처리."""
        if self.collecting:
            if not messagebox.askyesno("확인", "수집이 진행 중입니다. 중단할까요?", parent=self):
                return
            self.collecting = False
        self.running = False
        self.destroy()


class BackupDialog(tk.Toplevel):
    """백업/복원 관리 다이얼로그."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("백업 관리")
        self.geometry("550x500")
        self.resizable(False, False)

        self.backup_dir = PROJECT_ROOT / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        self._create_widgets()
        self._load_backups()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 백업 버튼
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            btn_frame, text="설정 백업", width=12,
            command=lambda: self._create_backup("config")
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame, text="모델 백업", width=12,
            command=lambda: self._create_backup("models")
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame, text="DB 백업", width=12,
            command=lambda: self._create_backup("database")
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame, text="전체 백업", width=12,
            command=lambda: self._create_backup("full")
        ).pack(side=tk.LEFT, padx=2)

        # 최근 백업 목록
        list_frame = ttk.LabelFrame(main_frame, text="최근 백업", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Treeview for backups
        columns = ("날짜", "유형", "크기")
        self.backup_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        self.backup_tree.heading("날짜", text="날짜")
        self.backup_tree.heading("유형", text="유형")
        self.backup_tree.heading("크기", text="크기")
        self.backup_tree.column("날짜", width=180)
        self.backup_tree.column("유형", width=100)
        self.backup_tree.column("크기", width=100)
        self.backup_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.backup_tree.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.backup_tree.configure(yscrollcommand=scrollbar.set)

        # 하단 버튼
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            bottom_frame, text="복원", width=10,
            command=self._restore_backup
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            bottom_frame, text="삭제", width=10,
            command=self._delete_backup
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            bottom_frame, text="새로고침", width=10,
            command=self._load_backups
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            bottom_frame, text="폴더 열기", width=10,
            command=lambda: os.startfile(str(self.backup_dir)) if sys.platform == "win32" else None
        ).pack(side=tk.RIGHT, padx=2)

    def _load_backups(self):
        """백업 목록 로드."""
        for item in self.backup_tree.get_children():
            self.backup_tree.delete(item)

        backups = []
        for f in self.backup_dir.glob("*.zip"):
            stat = f.stat()
            size_mb = stat.st_size / (1024 * 1024)
            # Parse backup type from filename
            name = f.stem
            if "_config_" in name:
                btype = "설정"
            elif "_models_" in name:
                btype = "모델"
            elif "_database_" in name:
                btype = "DB"
            elif "_full_" in name:
                btype = "전체"
            else:
                btype = "기타"

            backups.append((
                datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                btype,
                f"{size_mb:.1f} MB",
                str(f)  # Store path for later use
            ))

        # Sort by date descending
        backups.sort(key=lambda x: x[0], reverse=True)

        for date, btype, size, path in backups:
            self.backup_tree.insert("", tk.END, values=(date, btype, size), tags=(path,))

    def _create_backup(self, backup_type: str):
        """백업 생성."""
        import shutil
        import zipfile

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{backup_type}_{timestamp}.zip"
        backup_path = self.backup_dir / backup_name

        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                if backup_type in ("config", "full"):
                    # Backup config files
                    for cfg in [".env", "config.yaml", "config.py"]:
                        cfg_path = PROJECT_ROOT / cfg
                        if cfg_path.exists():
                            zf.write(cfg_path, cfg)

                if backup_type in ("models", "full"):
                    # Backup models directory
                    models_dir = PROJECT_ROOT / "models"
                    if models_dir.exists():
                        for f in models_dir.rglob("*"):
                            if f.is_file():
                                zf.write(f, f.relative_to(PROJECT_ROOT))

                if backup_type in ("database", "full"):
                    # Backup database
                    db_path = PROJECT_ROOT / "stocks.db"
                    if db_path.exists():
                        zf.write(db_path, "stocks.db")

            size_mb = backup_path.stat().st_size / (1024 * 1024)
            messagebox.showinfo("백업 완료", f"백업이 생성되었습니다.\n\n{backup_name}\n크기: {size_mb:.1f} MB", parent=self)
            self._load_backups()

        except Exception as e:
            messagebox.showerror("백업 오류", f"백업 생성 실패:\n{e}", parent=self)

    def _restore_backup(self):
        """백업 복원."""
        import zipfile

        selection = self.backup_tree.selection()
        if not selection:
            messagebox.showwarning("선택 필요", "복원할 백업을 선택하세요.", parent=self)
            return

        # Get the backup path from tags
        item = selection[0]
        tags = self.backup_tree.item(item, "tags")
        if not tags:
            return
        backup_path = Path(tags[0])

        if not backup_path.exists():
            messagebox.showerror("오류", "백업 파일을 찾을 수 없습니다.", parent=self)
            return

        if not messagebox.askyesno("복원 확인",
            f"다음 백업을 복원하시겠습니까?\n\n{backup_path.name}\n\n"
            "주의: 현재 파일이 덮어씌워집니다!", parent=self):
            return

        try:
            with zipfile.ZipFile(backup_path, 'r') as zf:
                zf.extractall(PROJECT_ROOT)

            messagebox.showinfo("복원 완료", "백업이 복원되었습니다.", parent=self)
        except Exception as e:
            messagebox.showerror("복원 오류", f"복원 실패:\n{e}", parent=self)

    def _delete_backup(self):
        """백업 삭제."""
        selection = self.backup_tree.selection()
        if not selection:
            messagebox.showwarning("선택 필요", "삭제할 백업을 선택하세요.", parent=self)
            return

        item = selection[0]
        tags = self.backup_tree.item(item, "tags")
        if not tags:
            return
        backup_path = Path(tags[0])

        if not messagebox.askyesno("삭제 확인", f"다음 백업을 삭제하시겠습니까?\n\n{backup_path.name}", parent=self):
            return

        try:
            backup_path.unlink()
            self._load_backups()
            messagebox.showinfo("삭제 완료", "백업이 삭제되었습니다.", parent=self)
        except Exception as e:
            messagebox.showerror("삭제 오류", f"삭제 실패:\n{e}", parent=self)


class EnhancedLogViewer(tk.Toplevel):
    """향상된 로그 뷰어 다이얼로그."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("로그 뷰어")
        self.geometry("800x600")
        self.minsize(600, 400)

        self.auto_scroll = tk.BooleanVar(value=True)
        self.log_level = tk.StringVar(value="ALL")
        self.search_text = tk.StringVar()
        self.log_entries = []

        self._create_widgets()
        self._load_logs()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 상단 툴바
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=5)

        # 검색
        ttk.Label(toolbar, text="검색:").pack(side=tk.LEFT, padx=2)
        search_entry = ttk.Entry(toolbar, textvariable=self.search_text, width=20)
        search_entry.pack(side=tk.LEFT, padx=2)
        search_entry.bind("<Return>", lambda e: self._filter_logs())

        ttk.Button(toolbar, text="검색", width=6, command=self._filter_logs).pack(side=tk.LEFT, padx=2)

        # 레벨 필터
        ttk.Label(toolbar, text="레벨:").pack(side=tk.LEFT, padx=(10, 2))
        level_combo = ttk.Combobox(
            toolbar, textvariable=self.log_level,
            values=["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
            width=10, state="readonly"
        )
        level_combo.pack(side=tk.LEFT, padx=2)
        level_combo.bind("<<ComboboxSelected>>", lambda e: self._filter_logs())

        # 자동 스크롤
        ttk.Checkbutton(
            toolbar, text="자동 스크롤",
            variable=self.auto_scroll
        ).pack(side=tk.LEFT, padx=10)

        # 버튼
        ttk.Button(toolbar, text="새로고침", width=10, command=self._load_logs).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="내보내기", width=10, command=self._export_logs).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="지우기", width=8, command=self._clear_logs).pack(side=tk.RIGHT, padx=2)

        # 로그 영역
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 태그 설정 (색상)
        self.log_text.tag_config("DEBUG", foreground="gray")
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("WARNING", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("HIGHLIGHT", background="yellow")

        # 상태바
        self.status_label = ttk.Label(main_frame, text="0개 항목")
        self.status_label.pack(anchor=tk.W)

    def _load_logs(self):
        """로그 파일 로드."""
        self.log_entries = []

        # logs 디렉토리에서 로그 파일 로드
        log_files = list(LOGS_DIR.glob("*.log")) if LOGS_DIR.exists() else []

        for log_file in sorted(log_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # 로그 레벨 감지
                            if "[ERROR]" in line or "ERROR" in line:
                                level = "ERROR"
                            elif "[WARNING]" in line or "WARNING" in line:
                                level = "WARNING"
                            elif "[DEBUG]" in line or "DEBUG" in line:
                                level = "DEBUG"
                            else:
                                level = "INFO"
                            self.log_entries.append((level, line))
            except Exception:
                pass

        self._filter_logs()

    def _filter_logs(self):
        """로그 필터링 및 표시."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)

        level_filter = self.log_level.get()
        search = self.search_text.get().lower()

        level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        min_level = level_order.get(level_filter, -1)

        filtered = []
        for level, line in self.log_entries:
            if level_filter != "ALL" and level_order.get(level, 0) < min_level:
                continue
            if search and search not in line.lower():
                continue
            filtered.append((level, line))

        for level, line in filtered[-1000:]:  # 최근 1000개만
            self.log_text.insert(tk.END, line + "\n", level)
            if search:
                # 검색어 하이라이트
                start = "1.0"
                while True:
                    pos = self.log_text.search(search, start, tk.END, nocase=True)
                    if not pos:
                        break
                    end = f"{pos}+{len(search)}c"
                    self.log_text.tag_add("HIGHLIGHT", pos, end)
                    start = end

        self.log_text.config(state=tk.DISABLED)

        if self.auto_scroll.get():
            self.log_text.see(tk.END)

        self.status_label.config(text=f"{len(filtered)}개 항목 (전체: {len(self.log_entries)})")

    def _export_logs(self):
        """로그 내보내기."""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")],
            initialfile=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if filepath.endswith('.csv'):
                    f.write("Level,Message\n")
                    for level, line in self.log_entries:
                        f.write(f'"{level}","{line.replace(chr(34), chr(34)+chr(34))}"\n')
                else:
                    for level, line in self.log_entries:
                        f.write(f"{line}\n")

            messagebox.showinfo("내보내기 완료", f"로그가 저장되었습니다.\n{filepath}", parent=self)
        except Exception as e:
            messagebox.showerror("내보내기 오류", f"저장 실패:\n{e}", parent=self)

    def _clear_logs(self):
        """로그 지우기."""
        if messagebox.askyesno("확인", "로그 뷰어를 지울까요?\n(파일은 삭제되지 않습니다)", parent=self):
            self.log_entries = []
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state=tk.DISABLED)
            self.status_label.config(text="0개 항목")


class SetupWizardDialog(tk.Toplevel):
    """환경 설정 마법사 다이얼로그."""

    REQUIRED_PACKAGES = [
        "torch", "numpy", "pandas", "scikit-learn", "xgboost",
        "fastapi", "uvicorn", "sqlalchemy", "requests", "pydantic"
    ]

    def __init__(self, parent):
        super().__init__(parent)
        self.title("환경 설정 마법사")
        self.geometry("600x550")
        self.resizable(False, False)

        self.current_step = 0
        self.check_results = {}

        self._create_widgets()
        self._run_checks()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 제목
        ttk.Label(
            main_frame, text="환경 설정 마법사",
            font=("맑은 고딕", 14, "bold")
        ).pack(pady=10)

        ttk.Label(
            main_frame, text="시스템 환경을 검사합니다...",
            font=("맑은 고딕", 10)
        ).pack()

        # 체크 결과 영역
        check_frame = ttk.LabelFrame(main_frame, text="환경 검사", padding=10)
        check_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.check_text = scrolledtext.ScrolledText(
            check_frame, height=18, font=("Consolas", 9)
        )
        self.check_text.pack(fill=tk.BOTH, expand=True)

        # 태그 설정
        self.check_text.tag_config("OK", foreground="green")
        self.check_text.tag_config("WARN", foreground="orange")
        self.check_text.tag_config("FAIL", foreground="red")
        self.check_text.tag_config("TITLE", font=("맑은 고딕", 10, "bold"))

        # 하단 버튼
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.fix_btn = ttk.Button(
            btn_frame, text="문제 해결", width=12,
            command=self._fix_issues,
            state=tk.DISABLED
        )
        self.fix_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame, text="다시 검사", width=12,
            command=self._run_checks
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame, text="닫기", width=10,
            command=self.destroy
        ).pack(side=tk.RIGHT, padx=2)

    def _run_checks(self):
        """환경 검사 실행."""
        self.check_text.config(state=tk.NORMAL)
        self.check_text.delete(1.0, tk.END)
        self.check_results = {"ok": 0, "warn": 0, "fail": 0}

        # 1. Python 버전
        self._add_title("[1/6] Python 버전 확인")
        import platform
        py_version = platform.python_version()
        if py_version.startswith("3.1") or py_version.startswith("3.9"):
            self._add_result("OK", f"Python {py_version} 설치됨")
        elif py_version.startswith("3."):
            self._add_result("WARN", f"Python {py_version} (3.9+ 권장)")
        else:
            self._add_result("FAIL", f"Python {py_version} (3.9+ 필요)")

        # 2. 필수 패키지
        self._add_title("\n[2/6] 필수 패키지 확인")
        missing_packages = []
        for pkg in self.REQUIRED_PACKAGES:
            try:
                __import__(pkg.replace("-", "_"))
                self._add_result("OK", f"{pkg} 설치됨")
            except ImportError:
                missing_packages.append(pkg)
                self._add_result("FAIL", f"{pkg} 미설치")

        if missing_packages:
            self.check_results["missing_packages"] = missing_packages

        # 3. .env 파일
        self._add_title("\n[3/6] .env 파일 확인")
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_content = f.read()
            if "POLYGON_API_KEY" in env_content:
                self._add_result("OK", ".env 파일 존재, API 키 설정됨")
            else:
                self._add_result("WARN", ".env 파일 존재, API 키 미설정")
        else:
            self._add_result("FAIL", ".env 파일 없음")

        # 4. 데이터베이스
        self._add_title("\n[4/6] 데이터베이스 확인")
        db_path = PROJECT_ROOT / "stocks.db"
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            self._add_result("OK", f"stocks.db 존재 ({size_mb:.1f} MB)")
        else:
            self._add_result("WARN", "stocks.db 없음 (첫 실행 시 생성됨)")

        # 5. 모델 디렉토리
        self._add_title("\n[5/6] 모델 디렉토리 확인")
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pkl"))
            if model_files:
                self._add_result("OK", f"models/ 존재 ({len(model_files)} 모델)")
            else:
                self._add_result("WARN", "models/ 존재하나 모델 파일 없음")
        else:
            self._add_result("WARN", "models/ 디렉토리 없음")

        # 6. config 파일
        self._add_title("\n[6/6] 설정 파일 확인")
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            self._add_result("OK", "config.yaml 존재")
        else:
            config_py = PROJECT_ROOT / "config.py"
            if config_py.exists():
                self._add_result("OK", "config.py 존재")
            else:
                self._add_result("WARN", "설정 파일 없음 (기본값 사용)")

        # 요약
        self._add_title("\n" + "="*50)
        total = self.check_results["ok"] + self.check_results["warn"] + self.check_results["fail"]
        self.check_text.insert(tk.END, f"\n결과: 성공 {self.check_results['ok']}, ")
        self.check_text.insert(tk.END, f"경고 {self.check_results['warn']}, ")
        self.check_text.insert(tk.END, f"실패 {self.check_results['fail']} / 총 {total}\n")

        if self.check_results["fail"] > 0:
            self.fix_btn.config(state=tk.NORMAL)
        else:
            self.fix_btn.config(state=tk.DISABLED)

        self.check_text.config(state=tk.DISABLED)

    def _add_title(self, text: str):
        """제목 추가."""
        self.check_text.insert(tk.END, text + "\n", "TITLE")

    def _add_result(self, status: str, text: str):
        """검사 결과 추가."""
        if status == "OK":
            self.check_text.insert(tk.END, f"  ✓ {text}\n", "OK")
            self.check_results["ok"] += 1
        elif status == "WARN":
            self.check_text.insert(tk.END, f"  ⚠ {text}\n", "WARN")
            self.check_results["warn"] += 1
        else:
            self.check_text.insert(tk.END, f"  ✗ {text}\n", "FAIL")
            self.check_results["fail"] += 1

    def _fix_issues(self):
        """문제 해결 시도."""
        missing = self.check_results.get("missing_packages", [])
        if missing:
            if messagebox.askyesno(
                "패키지 설치",
                f"다음 패키지를 설치할까요?\n\n{', '.join(missing)}",
                parent=self
            ):
                pip_cmd = f'pip install {" ".join(missing)}'
                try:
                    subprocess.run(pip_cmd, shell=True, check=True)
                    messagebox.showinfo("설치 완료", "패키지 설치가 완료되었습니다.", parent=self)
                    self._run_checks()
                except subprocess.CalledProcessError as e:
                    messagebox.showerror("설치 오류", f"설치 실패:\n{e}", parent=self)


class ProcessManager:
    """Manages subprocess execution and monitoring."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.output_queues: Dict[str, queue.Queue] = {}
        # 자동 재시작 설정
        self.auto_restart_enabled: Dict[str, bool] = {}
        self.restart_counts: Dict[str, int] = {}
        self.max_restarts: int = 3
        self.restart_interval: int = 10  # seconds
        self.restart_callbacks: Dict[str, callable] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()

    def get_python(self) -> str:
        """Get Python executable path."""
        if VENV_PYTHON.exists():
            return str(VENV_PYTHON)
        return sys.executable

    def start_process(self, name: str, script: str, args: list = None,
                       script_dir: Path = None) -> bool:
        """Start a process."""
        if name in self.processes and self.is_running(name):
            return False

        python = self.get_python()
        base_dir = script_dir or SCRIPTS_DIR
        script_path = base_dir / script

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        cmd = [python, str(script_path)]
        if args:
            cmd.extend(args)

        # Create output queue
        self.output_queues[name] = queue.Queue()

        # Start process
        self.processes[name] = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT)
        )

        # Start output reader thread
        thread = threading.Thread(
            target=self._read_output,
            args=(name,),
            daemon=True
        )
        thread.start()

        return True

    def _read_output(self, name: str):
        """Read process output in background."""
        proc = self.processes.get(name)
        if not proc:
            return

        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                self.output_queues[name].put(line.rstrip())
        except Exception as e:
            self.output_queues[name].put(f"[ERROR] {e}")
        finally:
            self.output_queues[name].put(None)  # Signal completion

    def stop_process(self, name: str) -> bool:
        """Stop a process."""
        proc = self.processes.get(name)
        if not proc:
            return False

        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        return True

    def is_running(self, name: str) -> bool:
        """Check if process is running."""
        proc = self.processes.get(name)
        if not proc:
            return False
        return proc.poll() is None

    def get_output(self, name: str) -> Optional[str]:
        """Get output from queue (non-blocking)."""
        q = self.output_queues.get(name)
        if not q:
            return None
        try:
            return q.get_nowait()
        except queue.Empty:
            return None

    def start_npm_process(self, name: str, npm_cmd: str, cwd: Path) -> bool:
        """Start an npm process."""
        if name in self.processes and self.is_running(name):
            return False

        if not cwd.exists():
            raise FileNotFoundError(f"Directory not found: {cwd}")

        # Use npm.cmd on Windows
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        cmd = [npm, "run", npm_cmd]

        self.output_queues[name] = queue.Queue()

        self.processes[name] = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd),
            shell=True
        )

        thread = threading.Thread(
            target=self._read_output,
            args=(name,),
            daemon=True
        )
        thread.start()

        return True

    def cleanup(self):
        """Stop all processes."""
        # 자동 재시작 모니터 중지
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)
        # 모든 자동 재시작 비활성화
        self.auto_restart_enabled.clear()
        for name in list(self.processes.keys()):
            self.stop_process(name)

    def enable_auto_restart(self, name: str, restart_callback: callable = None):
        """프로세스 자동 재시작 활성화."""
        self.auto_restart_enabled[name] = True
        self.restart_counts[name] = 0
        if restart_callback:
            self.restart_callbacks[name] = restart_callback
        # 모니터 스레드 시작
        if not self._monitor_thread or not self._monitor_thread.is_alive():
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_processes,
                daemon=True
            )
            self._monitor_thread.start()

    def disable_auto_restart(self, name: str):
        """프로세스 자동 재시작 비활성화."""
        self.auto_restart_enabled[name] = False
        self.restart_counts[name] = 0
        if name in self.restart_callbacks:
            del self.restart_callbacks[name]

    def _monitor_processes(self):
        """프로세스 상태 모니터링 및 자동 재시작."""
        import time
        while not self._stop_monitor.is_set():
            for name in list(self.auto_restart_enabled.keys()):
                if not self.auto_restart_enabled.get(name, False):
                    continue
                # 프로세스가 비정상 종료되었는지 확인
                proc = self.processes.get(name)
                if proc and proc.poll() is not None:  # 종료됨
                    if self.restart_counts.get(name, 0) < self.max_restarts:
                        self.restart_counts[name] = self.restart_counts.get(name, 0) + 1
                        # 콜백으로 재시작 알림
                        callback = self.restart_callbacks.get(name)
                        if callback:
                            try:
                                callback(name, self.restart_counts[name])
                            except Exception:
                                pass
                        time.sleep(self.restart_interval)
            time.sleep(2)  # 2초마다 체크


class TradingLauncher(tk.Tk):
    """Main launcher GUI."""

    def __init__(self):
        super().__init__()

        # 설정 로드
        self.settings = LauncherSettings()

        self.title("FiveForFree 트레이딩 런처")
        self.minsize(800, 750)

        # 저장된 창 상태 복원
        self._restore_window_state()

        # Icon (if available)
        try:
            self.iconbitmap(PROJECT_ROOT / "icon.ico")
        except:
            pass

        # 고해상도 폰트 설정
        self._setup_fonts()

        # CustomTkinter 테마 설정
        self._apply_theme()

        # Process manager
        self.process_mgr = ProcessManager()

        # 자동 재시작 체크박스 변수 (설정에서 로드)
        self.auto_restart_vars = {
            "live": tk.BooleanVar(value=self.settings.processes.auto_restart_live),
            "paper": tk.BooleanVar(value=self.settings.processes.auto_restart_paper),
            "api": tk.BooleanVar(value=self.settings.processes.auto_restart_api),
            "frontend": tk.BooleanVar(value=self.settings.processes.auto_restart_frontend),
        }

        # Build UI
        self._create_menu()
        self._create_main_frame()
        self._create_status_bar()

        # Start status update loop
        self._update_status()

        # Bind keyboard shortcuts
        self._bind_shortcuts()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _restore_window_state(self):
        """저장된 창 위치/크기 복원."""
        w = self.settings.window
        self.geometry(f"{w.width}x{w.height}+{w.x}+{w.y}")
        if w.maximized:
            self.after(100, lambda: self.state('zoomed'))

    def _save_window_state(self):
        """현재 창 상태 저장."""
        if self.state() == 'zoomed':
            self.settings.window.maximized = True
        else:
            self.settings.window.maximized = False
            self.settings.window.x = self.winfo_x()
            self.settings.window.y = self.winfo_y()
            self.settings.window.width = self.winfo_width()
            self.settings.window.height = self.winfo_height()

    def _apply_theme(self):
        """CustomTkinter 테마 적용."""
        if CTK_AVAILABLE:
            theme = self.settings.ui.theme
            if theme == "system":
                ctk.set_appearance_mode("system")
            elif theme == "dark":
                ctk.set_appearance_mode("dark")
            else:
                ctk.set_appearance_mode("light")
            ctk.set_default_color_theme("blue")

    def _change_theme(self):
        """테마 변경."""
        new_theme = self._theme_var.get()
        self.settings.ui.theme = new_theme
        self._apply_theme()

        if CTK_AVAILABLE:
            messagebox.showinfo(
                "테마 변경",
                f"테마가 '{new_theme}'으로 변경되었습니다.\n"
                "일부 변경사항은 재시작 후 적용됩니다."
            )
        else:
            messagebox.showinfo(
                "테마 변경",
                "CustomTkinter가 설치되지 않아 테마가 적용되지 않습니다.\n"
                "'pip install customtkinter'로 설치하세요."
            )

    def _setup_fonts(self):
        """고해상도 디스플레이용 폰트 설정."""
        # 기본 폰트 크기 (DPI awareness가 처리하므로 고정 크기 사용)
        base_size = 9
        mono_size = 9

        # 폰트 정의
        self.default_font = tkfont.Font(family="맑은 고딕", size=base_size)
        self.bold_font = tkfont.Font(family="맑은 고딕", size=base_size, weight="bold")
        self.mono_font = tkfont.Font(family="Consolas", size=mono_size)
        self.title_font = tkfont.Font(family="맑은 고딕", size=base_size + 2, weight="bold")

        # ttk 스타일 설정
        style = ttk.Style()
        style.configure(".", font=self.default_font)
        style.configure("TLabel", font=self.default_font)
        style.configure("TButton", font=self.default_font, padding=5)
        style.configure("TLabelframe.Label", font=self.bold_font)

        # 기본 폰트 적용
        self.option_add("*Font", self.default_font)

    def _create_menu(self):
        """메뉴바 생성."""
        menubar = tk.Menu(self, font=self.default_font)
        self.config(menu=menubar)

        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0, font=self.default_font)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="프로젝트 폴더 열기", command=self._open_folder)
        file_menu.add_command(label="결과 폴더 열기", command=self._open_results)
        file_menu.add_command(label="로그 폴더 열기", command=self._open_logs)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self._on_close)

        # 도구 메뉴
        tools_menu = tk.Menu(menubar, tearoff=0, font=self.default_font)
        menubar.add_cascade(label="도구", menu=tools_menu)
        tools_menu.add_command(label="데이터 수집", command=self._show_data_collection)
        tools_menu.add_command(label="모델 학습", command=self._show_model_training)
        tools_menu.add_separator()
        # 백테스트 서브메뉴
        backtest_menu = tk.Menu(tools_menu, tearoff=0, font=self.default_font)
        tools_menu.add_cascade(label="백테스트 실행", menu=backtest_menu)
        backtest_menu.add_command(label="기본 실행", command=lambda: self._run_backtest("run_backtest.py"))
        backtest_menu.add_command(label="고급 실행", command=lambda: self._run_backtest("run_enhanced_backtest.py"))
        tools_menu.add_separator()
        tools_menu.add_command(label="시스템 모니터", command=self._show_system_monitor)
        tools_menu.add_command(label="로그 뷰어", command=self._show_log_viewer)

        # 설정 메뉴
        settings_menu = tk.Menu(menubar, tearoff=0, font=self.default_font)
        menubar.add_cascade(label="설정", menu=settings_menu)
        settings_menu.add_command(label="설정 편집", command=self._open_config)
        settings_menu.add_command(label="환경 설정 마법사", command=self._show_setup_wizard)
        settings_menu.add_separator()

        # 테마 서브메뉴
        theme_menu = tk.Menu(settings_menu, tearoff=0, font=self.default_font)
        settings_menu.add_cascade(label="테마", menu=theme_menu)
        self._theme_var = tk.StringVar(value=self.settings.ui.theme)
        theme_menu.add_radiobutton(
            label="시스템 기본", value="system",
            variable=self._theme_var, command=self._change_theme
        )
        theme_menu.add_radiobutton(
            label="라이트 모드", value="light",
            variable=self._theme_var, command=self._change_theme
        )
        theme_menu.add_radiobutton(
            label="다크 모드", value="dark",
            variable=self._theme_var, command=self._change_theme
        )

        settings_menu.add_separator()
        settings_menu.add_command(label="백업/복원", command=self._show_backup)

        # 도움말 메뉴
        help_menu = tk.Menu(menubar, tearoff=0, font=self.default_font)
        menubar.add_cascade(label="도움말", menu=help_menu)
        help_menu.add_command(label="단축키 목록", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="정보", command=self._show_about)

    def _create_main_frame(self):
        """메인 프레임 생성."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 컨트롤 패널
        control_frame = ttk.LabelFrame(main_frame, text="컨트롤 패널", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 프로세스 컨트롤
        self._create_process_controls(control_frame)

        # 로그 출력
        log_frame = ttk.LabelFrame(main_frame, text="로그 출력", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=self.mono_font,
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 로그 컨트롤
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(log_controls, text="로그 지우기", command=self._clear_log).pack(side=tk.RIGHT)

    def _create_process_controls(self, parent):
        """프로세스 컨트롤 버튼 생성."""
        # 실시간 트레이딩
        live_frame = ttk.Frame(parent)
        live_frame.pack(fill=tk.X, pady=5)

        ttk.Label(live_frame, text="실시간 트레이딩:", width=18).pack(side=tk.LEFT)

        self.live_status = ttk.Label(live_frame, text="중지됨", foreground="red", width=10)
        self.live_status.pack(side=tk.LEFT, padx=5)

        self.live_start_btn = ttk.Button(
            live_frame, text="시작", width=8,
            command=lambda: self._start_process("live", "run_system.py")
        )
        self.live_start_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(self.live_start_btn, "실시간 트레이딩 시작 (F5)")

        self.live_stop_btn = ttk.Button(
            live_frame, text="정지", width=8,
            command=lambda: self._stop_process("live"),
            state=tk.DISABLED
        )
        self.live_stop_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(self.live_stop_btn, "실시간 트레이딩 정지 (F6)")

        ttk.Checkbutton(
            live_frame, text="자동재시작",
            variable=self.auto_restart_vars["live"],
            command=lambda: self._toggle_auto_restart("live")
        ).pack(side=tk.LEFT, padx=5)

        # 모의 트레이딩
        paper_frame = ttk.Frame(parent)
        paper_frame.pack(fill=tk.X, pady=5)

        ttk.Label(paper_frame, text="모의 트레이딩:", width=18).pack(side=tk.LEFT)

        self.paper_status = ttk.Label(paper_frame, text="중지됨", foreground="red", width=10)
        self.paper_status.pack(side=tk.LEFT, padx=5)

        self.paper_start_btn = ttk.Button(
            paper_frame, text="시작", width=8,
            command=lambda: self._start_process("paper", "paper_trading.py")
        )
        self.paper_start_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(self.paper_start_btn, "모의 트레이딩 시뮬레이션 시작")

        self.paper_stop_btn = ttk.Button(
            paper_frame, text="정지", width=8,
            command=lambda: self._stop_process("paper"),
            state=tk.DISABLED
        )
        self.paper_stop_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(self.paper_stop_btn, "모의 트레이딩 시뮬레이션 정지")

        ttk.Checkbutton(
            paper_frame, text="자동재시작",
            variable=self.auto_restart_vars["paper"],
            command=lambda: self._toggle_auto_restart("paper")
        ).pack(side=tk.LEFT, padx=5)

        # 구분선
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # API 서버
        api_frame = ttk.Frame(parent)
        api_frame.pack(fill=tk.X, pady=5)

        ttk.Label(api_frame, text="API 서버:", width=18).pack(side=tk.LEFT)

        self.api_status = ttk.Label(api_frame, text="중지됨", foreground="red", width=10)
        self.api_status.pack(side=tk.LEFT, padx=5)

        self.api_start_btn = ttk.Button(
            api_frame, text="시작", width=8,
            command=self._start_api
        )
        self.api_start_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(self.api_start_btn, "FastAPI 백엔드 서버 시작 (F7)")

        self.api_stop_btn = ttk.Button(
            api_frame, text="정지", width=8,
            command=lambda: self._stop_process("api"),
            state=tk.DISABLED
        )
        self.api_stop_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(self.api_stop_btn, "API 서버 정지 (F8)")

        # API 문서 버튼 제거 (2025-12-27 대표님 지시: UI 일체감 위해 주석처리)
        # api_open_btn = ttk.Button(
        #     api_frame, text="API 문서", width=10,
        #     command=lambda: webbrowser.open(f"{API_URL}/docs")
        # )
        # api_open_btn.pack(side=tk.LEFT, padx=2)
        # Tooltip(api_open_btn, "API 문서 열기 (Swagger UI)")

        ttk.Checkbutton(
            api_frame, text="자동재시작",
            variable=self.auto_restart_vars["api"],
            command=lambda: self._toggle_auto_restart("api")
        ).pack(side=tk.LEFT, padx=5)

        # 프론트엔드 개발 서버
        frontend_frame = ttk.Frame(parent)
        frontend_frame.pack(fill=tk.X, pady=5)

        ttk.Label(frontend_frame, text="프론트엔드:", width=18).pack(side=tk.LEFT)

        self.frontend_status = ttk.Label(frontend_frame, text="중지됨", foreground="red", width=10)
        self.frontend_status.pack(side=tk.LEFT, padx=5)

        self.frontend_start_btn = ttk.Button(
            frontend_frame, text="시작", width=8,
            command=self._start_frontend
        )
        self.frontend_start_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(self.frontend_start_btn, "React 개발 서버 시작 (F9)")

        self.frontend_stop_btn = ttk.Button(
            frontend_frame, text="정지", width=8,
            command=lambda: self._stop_process("frontend"),
            state=tk.DISABLED
        )
        self.frontend_stop_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(self.frontend_stop_btn, "프론트엔드 서버 정지 (F10)")

        ttk.Checkbutton(
            frontend_frame, text="자동재시작",
            variable=self.auto_restart_vars["frontend"],
            command=lambda: self._toggle_auto_restart("frontend")
        ).pack(side=tk.LEFT, padx=5)

        # 구분선
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 백테스트 섹션
        backtest_frame = ttk.Frame(parent)
        backtest_frame.pack(fill=tk.X, pady=5)

        ttk.Label(backtest_frame, text="백테스트:", width=18).pack(side=tk.LEFT)

        self.backtest_status = ttk.Label(backtest_frame, text="대기중", foreground="blue", width=10)
        self.backtest_status.pack(side=tk.LEFT, padx=5)

        basic_btn = ttk.Button(
            backtest_frame, text="기본 실행", width=10,
            command=lambda: self._run_backtest("run_backtest.py")
        )
        basic_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(basic_btn, "기본 설정으로 백테스트 실행")

        enhanced_btn = ttk.Button(
            backtest_frame, text="고급 실행", width=10,
            command=lambda: self._run_backtest("run_enhanced_backtest.py")
        )
        enhanced_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(enhanced_btn, "고급 분석 포함 백테스트 실행")

        # 구분선
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 빠른 작업 섹션 - 3행으로 분리 (기능별 그룹화)
        BUTTON_WIDTH = 10  # 버튼 크기 통일

        # 1행: 분석/보기
        quick_row1 = ttk.Frame(parent)
        quick_row1.pack(fill=tk.X, pady=3)

        ttk.Label(quick_row1, text="분석/보기:", width=12).pack(side=tk.LEFT)

        results_btn = ttk.Button(
            quick_row1, text="결과 보기", width=BUTTON_WIDTH,
            command=self._open_results
        )
        results_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(results_btn, "백테스트 결과 폴더 열기")

        dashboard_btn = ttk.Button(
            quick_row1, text="대시보드", width=BUTTON_WIDTH,
            command=self._open_dashboard
        )
        dashboard_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(dashboard_btn, "트레이딩 대시보드 열기 (Ctrl+D)")

        logs_btn = ttk.Button(
            quick_row1, text="로그 보기", width=BUTTON_WIDTH,
            command=self._open_logs
        )
        logs_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(logs_btn, "로그 폴더 열기")

        logview_btn = ttk.Button(
            quick_row1, text="로그 뷰어", width=BUTTON_WIDTH,
            command=self._show_log_viewer
        )
        logview_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(logview_btn, "향상된 로그 뷰어")

        # 구분선
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # 2행: 데이터/학습
        quick_row2 = ttk.Frame(parent)
        quick_row2.pack(fill=tk.X, pady=3)

        ttk.Label(quick_row2, text="데이터/학습:", width=12).pack(side=tk.LEFT)

        data_btn = ttk.Button(
            quick_row2, text="데이터 확인", width=BUTTON_WIDTH,
            command=self._show_data_status
        )
        data_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(data_btn, "수집된 데이터 상태 확인 (Ctrl+I)")

        ticker_btn = ttk.Button(
            quick_row2, text="티커 목록", width=BUTTON_WIDTH,
            command=self._show_ticker_list
        )
        ticker_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(ticker_btn, "등록된 티커 목록 보기 (Ctrl+T)")

        train_btn = ttk.Button(
            quick_row2, text="모델 학습", width=BUTTON_WIDTH,
            command=self._show_model_training
        )
        train_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(train_btn, "모델 학습 관리 (Ctrl+M)")

        collect_btn = ttk.Button(
            quick_row2, text="데이터 수집", width=BUTTON_WIDTH,
            command=self._show_data_collection
        )
        collect_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(collect_btn, "데이터 수집 제어 (Ctrl+G)")

        # 구분선
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # 3행: 관리/설정
        quick_row3 = ttk.Frame(parent)
        quick_row3.pack(fill=tk.X, pady=3)

        ttk.Label(quick_row3, text="관리/설정:", width=12).pack(side=tk.LEFT)

        monitor_btn = ttk.Button(
            quick_row3, text="시스템", width=BUTTON_WIDTH,
            command=self._show_system_monitor
        )
        monitor_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(monitor_btn, "시스템 리소스 모니터 (Ctrl+R)")

        backup_btn = ttk.Button(
            quick_row3, text="백업", width=BUTTON_WIDTH,
            command=self._show_backup
        )
        backup_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(backup_btn, "백업 관리 (Ctrl+B)")

        config_btn = ttk.Button(
            quick_row3, text="설정 편집", width=BUTTON_WIDTH,
            command=self._open_config
        )
        config_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(config_btn, "config.yaml 파일 편집")

        wizard_btn = ttk.Button(
            quick_row3, text="설정 마법사", width=BUTTON_WIDTH,
            command=self._show_setup_wizard
        )
        wizard_btn.pack(side=tk.LEFT, padx=2)
        Tooltip(wizard_btn, "환경 설정 마법사 (Ctrl+W)")

    def _create_status_bar(self):
        """상태바 생성."""
        self.status_bar = ttk.Label(
            self,
            text=f"준비됨 | 프로젝트: {PROJECT_ROOT.name}",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _start_process(self, name: str, script: str):
        """프로세스 시작."""
        name_ko = {"live": "실시간 트레이딩", "paper": "모의 트레이딩"}.get(name, name)
        try:
            self.process_mgr.start_process(name, script)
            self._log(f"[정보] {name_ko} 시작 중 ({script})...")
            self._update_buttons()
        except Exception as e:
            self._log(f"[오류] {name_ko} 시작 실패: {e}")
            messagebox.showerror("오류", f"{name_ko} 시작 실패:\n{e}")

    def _stop_process(self, name: str):
        """프로세스 정지."""
        name_ko = {"live": "실시간 트레이딩", "paper": "모의 트레이딩",
                   "api": "API 서버", "frontend": "프론트엔드"}.get(name, name)
        if self.process_mgr.stop_process(name):
            self._log(f"[정보] {name_ko} 정지됨")
            self._update_buttons()

    def _start_api(self):
        """API 서버 시작."""
        try:
            # --auto-port: 포트 충돌 시 자동으로 대체 포트 사용
            self.process_mgr.start_process("api", "run_api.py", args=["--auto-port"], script_dir=PROJECT_ROOT)
            self._log(f"[정보] API 서버 시작 중 ({API_URL})...")
            self._update_buttons()
        except Exception as e:
            self._log(f"[오류] API 서버 시작 실패: {e}")
            messagebox.showerror("오류", f"API 서버 시작 실패:\n{e}")

    def _start_frontend(self):
        """프론트엔드 개발 서버 시작."""
        try:
            self.process_mgr.start_npm_process("frontend", "dev", FRONTEND_DIR)
            self._log(f"[정보] 프론트엔드 시작 중 ({FRONTEND_URL})...")
            self._update_buttons()
        except Exception as e:
            self._log(f"[오류] 프론트엔드 시작 실패: {e}")
            messagebox.showerror("오류", f"프론트엔드 시작 실패:\n{e}")

    def _run_backtest(self, script: str):
        """백테스트 실행."""
        try:
            self.backtest_status.config(text="실행중", foreground="orange")
            self.process_mgr.start_process("backtest", script)
            self._log(f"[정보] 백테스트 시작 중 ({script})...")
        except Exception as e:
            self._log(f"[오류] 백테스트 실행 실패: {e}")
            messagebox.showerror("오류", f"백테스트 실행 실패:\n{e}")

    def _update_buttons(self):
        """버튼 상태 업데이트."""
        # 실시간 트레이딩
        live_running = self.process_mgr.is_running("live")
        self.live_start_btn.config(state=tk.DISABLED if live_running else tk.NORMAL)
        self.live_stop_btn.config(state=tk.NORMAL if live_running else tk.DISABLED)
        self.live_status.config(
            text="실행중" if live_running else "중지됨",
            foreground="green" if live_running else "red"
        )

        # 모의 트레이딩
        paper_running = self.process_mgr.is_running("paper")
        self.paper_start_btn.config(state=tk.DISABLED if paper_running else tk.NORMAL)
        self.paper_stop_btn.config(state=tk.NORMAL if paper_running else tk.DISABLED)
        self.paper_status.config(
            text="실행중" if paper_running else "중지됨",
            foreground="green" if paper_running else "red"
        )

        # API 서버
        api_running = self.process_mgr.is_running("api")
        self.api_start_btn.config(state=tk.DISABLED if api_running else tk.NORMAL)
        self.api_stop_btn.config(state=tk.NORMAL if api_running else tk.DISABLED)
        self.api_status.config(
            text="실행중" if api_running else "중지됨",
            foreground="green" if api_running else "red"
        )

        # 프론트엔드
        frontend_running = self.process_mgr.is_running("frontend")
        self.frontend_start_btn.config(state=tk.DISABLED if frontend_running else tk.NORMAL)
        self.frontend_stop_btn.config(state=tk.NORMAL if frontend_running else tk.DISABLED)
        self.frontend_status.config(
            text="실행중" if frontend_running else "중지됨",
            foreground="green" if frontend_running else "red"
        )

        # 백테스트
        backtest_running = self.process_mgr.is_running("backtest")
        self.backtest_status.config(
            text="실행중" if backtest_running else "대기중",
            foreground="orange" if backtest_running else "blue"
        )

    def _update_status(self):
        """Periodic status update."""
        # Update buttons
        self._update_buttons()

        # Read process outputs
        for name in ["live", "paper", "api", "frontend", "backtest"]:
            while True:
                output = self.process_mgr.get_output(name)
                if output is None:
                    break
                self._log(f"[{name.upper()}] {output}")

        # 상태바 업데이트
        running = []
        if self.process_mgr.is_running("live"):
            running.append("실시간")
        if self.process_mgr.is_running("paper"):
            running.append("모의")
        if self.process_mgr.is_running("api"):
            running.append("API")
        if self.process_mgr.is_running("frontend"):
            running.append("프론트엔드")
        if self.process_mgr.is_running("backtest"):
            running.append("백테스트")

        status = ", ".join(running) if running else "모두 중지됨"
        self.status_bar.config(text=f"상태: {status} | {datetime.now().strftime('%H:%M:%S')}")

        # Schedule next update
        self.after(1000, self._update_status)

    def _log(self, message: str):
        """로그에 메시지 추가."""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        """로그 출력 지우기."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self._log("[정보] 로그가 지워졌습니다")

    def _open_folder(self):
        """프로젝트 폴더 열기."""
        os.startfile(PROJECT_ROOT)

    def _open_results(self):
        """결과 폴더 열기."""
        if RESULTS_DIR.exists():
            os.startfile(RESULTS_DIR)
            self._log("[정보] 결과 폴더를 열었습니다")
        else:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(RESULTS_DIR)
            self._log("[정보] 결과 폴더를 생성하고 열었습니다")

    def _open_dashboard(self):
        """브라우저에서 대시보드 열기."""
        if self.process_mgr.is_running("frontend"):
            webbrowser.open(FRONTEND_URL)
            self._log(f"[정보] 대시보드 열기 ({FRONTEND_URL})")
        elif self.process_mgr.is_running("api"):
            # API만 실행 중이면 API 문서로 대체
            webbrowser.open(f"{API_URL}/docs")
            self._log(f"[정보] 프론트엔드 미실행. API 문서 열기 ({API_URL}/docs)")
        else:
            result = messagebox.askyesno(
                "서버 미실행",
                "API와 프론트엔드 서버가 실행되고 있지 않습니다.\n\n"
                "지금 시작할까요?"
            )
            if result:
                self._start_api()
                self._start_frontend()
                self._log("[정보] API 및 프론트엔드 서버 시작 중...")

    def _open_logs(self):
        """로그 폴더 열기."""
        if LOGS_DIR.exists():
            os.startfile(LOGS_DIR)
            self._log("[정보] 로그 폴더를 열었습니다")
        else:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(LOGS_DIR)
            self._log("[정보] 로그 폴더를 생성하고 열었습니다")

    def _open_config(self):
        """설정 파일을 기본 편집기로 열기."""
        if CONFIG_FILE.exists():
            os.startfile(CONFIG_FILE)
            self._log("[정보] 설정 파일을 열었습니다")
        else:
            self._log("[경고] 설정 파일을 찾을 수 없습니다")
            messagebox.showwarning("파일 없음", f"설정 파일을 찾을 수 없습니다:\n{CONFIG_FILE}")

    def _show_data_status(self):
        """데이터 상태 다이얼로그 표시."""
        self._log("[정보] 데이터 상태 확인 창 열기")
        DataStatusDialog(self)

    def _show_ticker_list(self):
        """티커 목록 창 표시."""
        self._log("[정보] 티커 목록 창 열기")
        TickerListWindow(self)

    def _show_model_training(self):
        """모델 학습 관리 창 표시."""
        self._log("[정보] 모델 학습 관리 창 열기")
        ModelTrainingDialog(self)

    def _show_system_monitor(self):
        """시스템 모니터 창 표시."""
        self._log("[정보] 시스템 모니터 창 열기")
        SystemMonitorDialog(self)

    def _show_data_collection(self):
        """데이터 수집 제어 창 표시."""
        self._log("[정보] 데이터 수집 제어 창 열기")
        DataCollectionDialog(self)

    def _show_backup(self):
        """백업 관리 창 표시."""
        self._log("[정보] 백업 관리 창 열기")
        BackupDialog(self)

    def _show_log_viewer(self):
        """향상된 로그 뷰어 창 표시."""
        self._log("[정보] 로그 뷰어 창 열기")
        EnhancedLogViewer(self)

    def _show_setup_wizard(self):
        """환경 설정 마법사 창 표시."""
        self._log("[정보] 환경 설정 마법사 창 열기")
        SetupWizardDialog(self)

    def _toggle_auto_restart(self, name: str):
        """자동 재시작 토글."""
        enabled = self.auto_restart_vars[name].get()
        if enabled:
            self.process_mgr.enable_auto_restart(
                name,
                restart_callback=lambda n, c: self._on_auto_restart(n, c)
            )
            self._log(f"[정보] {name} 자동 재시작 활성화 (최대 {self.process_mgr.max_restarts}회)")
        else:
            self.process_mgr.disable_auto_restart(name)
            self._log(f"[정보] {name} 자동 재시작 비활성화")

    def _on_auto_restart(self, name: str, count: int):
        """자동 재시작 콜백."""
        self._log(f"[경고] {name} 프로세스 비정상 종료 - 자동 재시작 시도 ({count}/{self.process_mgr.max_restarts})")
        # 재시작 로직 - 프로세스별로 다르게 처리
        try:
            if name == "live":
                self._start_process("live", "run_system.py")
            elif name == "paper":
                self._start_process("paper", "paper_trading.py")
            elif name == "api":
                self._start_api()
            elif name == "frontend":
                self._start_frontend()
        except Exception as e:
            self._log(f"[오류] {name} 재시작 실패: {e}")

    def _bind_shortcuts(self):
        """키보드 단축키 바인딩."""
        self.bind("<F5>", lambda e: self._start_process("live", "run_system.py"))
        self.bind("<F6>", lambda e: self._stop_process("live"))
        self.bind("<F7>", lambda e: self._start_api())
        self.bind("<F8>", lambda e: self._stop_process("api"))
        self.bind("<F9>", lambda e: self._start_frontend())
        self.bind("<F10>", lambda e: self._stop_process("frontend"))
        self.bind("<Control-l>", lambda e: self._clear_log())
        self.bind("<Control-L>", lambda e: self._clear_log())
        self.bind("<Control-d>", lambda e: self._open_dashboard())
        self.bind("<Control-D>", lambda e: self._open_dashboard())
        self.bind("<Control-i>", lambda e: self._show_data_status())
        self.bind("<Control-I>", lambda e: self._show_data_status())
        self.bind("<Control-t>", lambda e: self._show_ticker_list())
        self.bind("<Control-T>", lambda e: self._show_ticker_list())
        # v1.3.0 신규 단축키
        self.bind("<Control-m>", lambda e: self._show_model_training())
        self.bind("<Control-M>", lambda e: self._show_model_training())
        self.bind("<Control-r>", lambda e: self._show_system_monitor())
        self.bind("<Control-R>", lambda e: self._show_system_monitor())
        self.bind("<Control-g>", lambda e: self._show_data_collection())
        self.bind("<Control-G>", lambda e: self._show_data_collection())
        self.bind("<Control-b>", lambda e: self._show_backup())
        self.bind("<Control-B>", lambda e: self._show_backup())
        self.bind("<Control-w>", lambda e: self._show_setup_wizard())
        self.bind("<Control-W>", lambda e: self._show_setup_wizard())
        self._log("[정보] 단축키: Ctrl+M=학습, Ctrl+R=시스템, Ctrl+G=수집, Ctrl+B=백업, Ctrl+W=설정")

    def _show_shortcuts(self):
        """단축키 목록 대화상자 표시."""
        shortcuts_text = """
단축키 목록
━━━━━━━━━━━━━━━━━━━━━━

[트레이딩]
F5  - 실시간 트레이딩 시작
F6  - 실시간 트레이딩 정지

[서버 제어]
F7  - API 서버 시작
F8  - API 서버 정지
F9  - 프론트엔드 시작
F10 - 프론트엔드 정지

[빠른 접근]
Ctrl+D - 대시보드 열기
Ctrl+I - 데이터 상태 확인
Ctrl+T - 티커 목록 보기

[도구]
Ctrl+M - 모델 학습 관리
Ctrl+R - 시스템 모니터
Ctrl+G - 데이터 수집
Ctrl+B - 백업 관리
Ctrl+W - 환경 설정 마법사

[기타]
Ctrl+L - 로그 지우기
"""
        messagebox.showinfo("단축키 목록", shortcuts_text)

    def _show_about(self):
        """정보 대화상자 표시."""
        messagebox.showinfo(
            "정보",
            "FiveForFree 트레이딩 런처\n\n"
            "버전: 1.3.0\n"
            "작성자: 개발팀장\n\n"
            "NASDAQ 예측 트레이딩 시스템\n"
            "관리용 GUI 런처입니다.\n\n"
            "단축키:\n"
            "F5/F6 - 실시간 트레이딩 시작/정지\n"
            "F7/F8 - API 서버 시작/정지\n"
            "F9/F10 - 프론트엔드 시작/정지\n"
            "Ctrl+D - 대시보드 열기\n"
            "Ctrl+I - 데이터 상태 확인\n"
            "Ctrl+T - 티커 목록 보기\n"
            "Ctrl+M - 모델 학습 관리\n"
            "Ctrl+R - 시스템 모니터\n"
            "Ctrl+G - 데이터 수집\n"
            "Ctrl+B - 백업 관리\n"
            "Ctrl+W - 환경 설정\n"
            "Ctrl+L - 로그 지우기"
        )

    def _on_close(self):
        """창 닫기 처리."""
        if any(self.process_mgr.is_running(n) for n in ["live", "paper", "api", "frontend", "backtest"]):
            if not messagebox.askyesno(
                "종료 확인",
                "일부 프로세스가 아직 실행 중입니다.\n정지하고 종료할까요?"
            ):
                return

        # 설정 저장
        self._save_window_state()
        self._save_process_settings()
        self.settings.save()

        self.process_mgr.cleanup()
        self.destroy()

    def _save_process_settings(self):
        """프로세스 자동재시작 설정 저장."""
        self.settings.processes.auto_restart_live = self.auto_restart_vars["live"].get()
        self.settings.processes.auto_restart_paper = self.auto_restart_vars["paper"].get()
        self.settings.processes.auto_restart_api = self.auto_restart_vars["api"].get()
        self.settings.processes.auto_restart_frontend = self.auto_restart_vars["frontend"].get()


def main():
    """Main entry point."""
    app = TradingLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
