# Alerts 시스템 설계서

**작성자**: 분석팀장
**작성일**: 2025-12-23
**대상**: 개발팀장, 비서실장
**우선순위**: 중간

---

## 1. 개요

### 1.1 목적
FiveForFree 대시보드에 실시간 알림 시스템을 추가하여 사용자가 중요한 이벤트를 즉시 인지할 수 있도록 함.

### 1.2 현재 아키텍처

```
┌─────────────┐     WebSocket      ┌─────────────┐
│   Frontend  │◄──────────────────►│   Backend   │
│  (React +   │    REST API        │  (FastAPI)  │
│   Zustand)  │◄──────────────────►│             │
└─────────────┘                    └─────────────┘
```

**기존 실시간 기능**:
- `websocket.py`: ConnectionManager로 실시간 broadcast
- `prediction_update`: 예측 결과 실시간 전송
- `price_update`: 가격 데이터 실시간 전송
- `heartbeat`: 연결 상태 확인

---

## 2. 알림 유형 정의

### 2.1 가격 알림 (Price Alert)

| 항목 | 설명 |
|------|------|
| **트리거** | 지정 가격 도달 |
| **조건** | 이상/이하/도달 |
| **데이터** | ticker, target_price, current_price, direction |

```python
@dataclass
class PriceAlert:
    id: str
    user_id: str
    ticker: str
    target_price: float
    condition: Literal['above', 'below', 'equals']
    enabled: bool = True
    triggered_at: Optional[datetime] = None

    def check(self, current_price: float) -> bool:
        if self.condition == 'above':
            return current_price >= self.target_price
        elif self.condition == 'below':
            return current_price <= self.target_price
        else:  # equals (with tolerance)
            return abs(current_price - self.target_price) < 0.01
```

**사용 시나리오**:
- "AAPL이 $200 이상이 되면 알림"
- "NVDA가 $500 이하로 떨어지면 알림"

---

### 2.2 모델 신호 알림 (Signal Alert)

| 항목 | 설명 |
|------|------|
| **트리거** | 새 매수/매도 신호 발생 |
| **조건** | 신호 유형 + 최소 확률 |
| **데이터** | ticker, signal_type, probability, model_name |

```python
@dataclass
class SignalAlert:
    id: str
    user_id: str
    ticker: Optional[str]  # None = 모든 티커
    signal_type: Literal['buy', 'sell', 'both']
    min_probability: float = 0.60
    min_confidence: Literal['low', 'medium', 'high'] = 'medium'
    enabled: bool = True

    def check(self, prediction: PredictionResult) -> bool:
        # 확률 조건 확인
        if self.signal_type == 'buy':
            return prediction.up_probability >= self.min_probability
        elif self.signal_type == 'sell':
            return prediction.down_probability >= self.min_probability
        else:  # both
            return max(prediction.up_probability,
                      prediction.down_probability) >= self.min_probability
```

**사용 시나리오**:
- "TSLA 매수 신호 (60% 이상) 발생 시 알림"
- "모든 종목 매도 신호 (High Confidence) 발생 시 알림"

---

### 2.3 포트폴리오 알림 (Portfolio Alert)

| 항목 | 설명 |
|------|------|
| **트리거** | 손익 한도 도달 |
| **조건** | 손익률/금액 기준 |
| **데이터** | portfolio_id, pnl_amount, pnl_percent |

```python
@dataclass
class PortfolioAlert:
    id: str
    user_id: str
    alert_type: Literal['profit_target', 'stop_loss', 'daily_limit']
    threshold_percent: Optional[float] = None
    threshold_amount: Optional[float] = None
    enabled: bool = True

    def check(self, portfolio: Portfolio) -> bool:
        if self.alert_type == 'profit_target':
            return portfolio.unrealized_pnl_pct >= self.threshold_percent
        elif self.alert_type == 'stop_loss':
            return portfolio.unrealized_pnl_pct <= -abs(self.threshold_percent)
        elif self.alert_type == 'daily_limit':
            return portfolio.daily_pnl_pct <= -abs(self.threshold_percent)
        return False
```

**사용 시나리오**:
- "포트폴리오 수익 10% 도달 시 알림"
- "일일 손실 3% 초과 시 알림"

---

## 3. 알림 채널 분석

### 3.1 브라우저 알림 (Push Notification)

| 항목 | 내용 |
|------|------|
| **구현 복잡도** | 낮음 |
| **장점** | 즉각적, 별도 서비스 불필요 |
| **단점** | 브라우저 열려있어야 함, 권한 필요 |
| **기술 스택** | Web Notification API, Service Worker |

#### 구현 방법

**Frontend (React)**:
```javascript
// src/services/notifications.js
export class NotificationService {
  static async requestPermission() {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission();
      return permission === 'granted';
    }
    return false;
  }

  static show(title, options = {}) {
    if (Notification.permission === 'granted') {
      return new Notification(title, {
        icon: '/logo.png',
        badge: '/badge.png',
        vibrate: [200, 100, 200],
        ...options
      });
    }
  }

  static showPriceAlert(ticker, price, direction) {
    this.show(`${ticker} 가격 알림`, {
      body: `${ticker}이(가) $${price}에 도달했습니다. (${direction})`,
      tag: `price-${ticker}`,
      requireInteraction: true
    });
  }

  static showSignalAlert(ticker, signal, probability) {
    const signalText = signal === 'buy' ? '매수' : '매도';
    this.show(`${ticker} ${signalText} 신호`, {
      body: `${ticker} ${signalText} 신호 발생 (${(probability * 100).toFixed(1)}%)`,
      tag: `signal-${ticker}`,
      requireInteraction: true
    });
  }
}
```

**Zustand Store 추가**:
```javascript
// src/stores/alertStore.js
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useAlertStore = create(
  persist(
    (set, get) => ({
      // 알림 설정
      priceAlerts: [],
      signalAlerts: [],
      portfolioAlerts: [],

      // 알림 권한
      notificationPermission: 'default',

      // Actions
      addPriceAlert: (alert) => set((state) => ({
        priceAlerts: [...state.priceAlerts, { ...alert, id: Date.now().toString() }]
      })),

      removePriceAlert: (id) => set((state) => ({
        priceAlerts: state.priceAlerts.filter(a => a.id !== id)
      })),

      addSignalAlert: (alert) => set((state) => ({
        signalAlerts: [...state.signalAlerts, { ...alert, id: Date.now().toString() }]
      })),

      setNotificationPermission: (permission) => set({ notificationPermission: permission }),
    }),
    { name: 'alert-settings' }
  )
)
```

---

### 3.2 이메일 알림

| 항목 | 내용 |
|------|------|
| **구현 복잡도** | 중간 |
| **장점** | 기록 보존, 오프라인 확인 가능 |
| **단점** | 지연 가능, 스팸 필터 위험 |
| **기술 스택** | SMTP / SendGrid / AWS SES |

#### 구현 방법

**Backend 추가**:
```python
# src/notifications/email_sender.py
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from typing import Optional
from jinja2 import Template

class EmailSender:
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.username = settings.SMTP_USERNAME
        self.password = settings.SMTP_PASSWORD
        self.from_email = settings.FROM_EMAIL

    def send_alert_email(
        self,
        to_email: str,
        subject: str,
        alert_type: str,
        alert_data: dict
    ) -> bool:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email

            # HTML 템플릿 렌더링
            html_content = self._render_template(alert_type, alert_data)
            msg.attach(MIMEText(html_content, 'html'))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def _render_template(self, alert_type: str, data: dict) -> str:
        templates = {
            'price': '''
                <h2>가격 알림: {{ ticker }}</h2>
                <p>{{ ticker }}이(가) 목표가 ${{ target_price }}에 도달했습니다.</p>
                <p>현재가: ${{ current_price }}</p>
            ''',
            'signal': '''
                <h2>매매 신호: {{ ticker }}</h2>
                <p>{{ signal_type }} 신호가 발생했습니다.</p>
                <p>확률: {{ probability }}%</p>
            ''',
        }
        template = Template(templates.get(alert_type, ''))
        return template.render(**data)
```

**settings.py 추가**:
```python
# Email Settings
SMTP_HOST: str = "smtp.gmail.com"
SMTP_PORT: int = 587
SMTP_USERNAME: str = ""  # 환경변수로 관리
SMTP_PASSWORD: str = ""  # 환경변수로 관리
FROM_EMAIL: str = "alerts@fiveforfree.com"
```

---

### 3.3 Telegram 연동

| 항목 | 내용 |
|------|------|
| **구현 복잡도** | 중간 |
| **장점** | 즉각적, 모바일 알림, 무료 |
| **단점** | Telegram 필요, Bot 설정 필요 |
| **기술 스택** | python-telegram-bot |

#### 구현 방법

**Backend 추가**:
```python
# src/notifications/telegram_sender.py
import httpx
from typing import Optional

class TelegramSender:
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    async def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "HTML"
    ) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": parse_mode
                    }
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def send_price_alert(
        self,
        chat_id: str,
        ticker: str,
        target_price: float,
        current_price: float
    ):
        text = f"""
<b>📊 가격 알림</b>

종목: <code>{ticker}</code>
목표가: ${target_price:.2f}
현재가: ${current_price:.2f}

{ticker}이(가) 목표가에 도달했습니다!
        """
        await self.send_message(chat_id, text)

    async def send_signal_alert(
        self,
        chat_id: str,
        ticker: str,
        signal_type: str,
        probability: float
    ):
        emoji = "🟢" if signal_type == "buy" else "🔴"
        signal_text = "매수" if signal_type == "buy" else "매도"

        text = f"""
<b>{emoji} {signal_text} 신호</b>

종목: <code>{ticker}</code>
신호: {signal_text}
확률: {probability:.1%}

새로운 매매 신호가 발생했습니다!
        """
        await self.send_message(chat_id, text)
```

**Bot 설정 가이드**:
1. @BotFather에서 새 Bot 생성
2. Bot Token 획득
3. 사용자가 Bot에 `/start` 메시지 전송
4. `getUpdates` API로 chat_id 획득
5. 사용자 설정에 chat_id 저장

---

### 3.4 Slack 연동

| 항목 | 내용 |
|------|------|
| **구현 복잡도** | 중간 |
| **장점** | 팀 협업, 채널별 알림 분류 |
| **단점** | Slack 필요, Webhook 설정 필요 |
| **기술 스택** | Slack Webhook / Slack SDK |

#### 구현 방법

```python
# src/notifications/slack_sender.py
import httpx
from typing import List, Dict

class SlackSender:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send_message(self, blocks: List[Dict]) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json={"blocks": blocks}
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False

    async def send_price_alert(
        self,
        ticker: str,
        target_price: float,
        current_price: float
    ):
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"📊 가격 알림: {ticker}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*종목:* {ticker}"},
                    {"type": "mrkdwn", "text": f"*목표가:* ${target_price:.2f}"},
                    {"type": "mrkdwn", "text": f"*현재가:* ${current_price:.2f}"},
                ]
            }
        ]
        await self.send_message(blocks)
```

---

## 4. 구현 복잡도 분석

### 4.1 프론트엔드 변경사항

| 파일 | 변경 내용 | 복잡도 |
|------|----------|--------|
| 신규: `src/stores/alertStore.js` | 알림 설정 상태 관리 | 중간 |
| 신규: `src/services/notifications.js` | Web Notification API | 낮음 |
| 신규: `src/components/AlertSettings.jsx` | 알림 설정 UI | 중간 |
| 신규: `src/components/AlertList.jsx` | 알림 목록 UI | 낮음 |
| 수정: `src/main.jsx` | WebSocket 메시지 핸들러 추가 | 낮음 |
| 수정: `src/services/api.js` | 알림 API 엔드포인트 추가 | 낮음 |

**총 예상 작업량**: 3~4일

---

### 4.2 백엔드 요구사항

| 파일 | 변경 내용 | 복잡도 |
|------|----------|--------|
| 신규: `src/notifications/` | 알림 발송 모듈 | 중간 |
| 신규: `src/api/routes/alerts.py` | 알림 CRUD API | 중간 |
| 신규: `src/models/alert.py` | 알림 데이터 모델 | 낮음 |
| 수정: `src/api/websocket.py` | 알림 broadcast 추가 | 낮음 |
| 수정: `config/settings.py` | 알림 설정 추가 | 낮음 |
| 신규: `src/workers/alert_worker.py` | 알림 체크 백그라운드 | 중간 |

**총 예상 작업량**: 4~5일

---

### 4.3 데이터 저장소

**옵션 1: SQLite (권장 - 단순 구현)**
```python
# src/models/alert.py
from sqlalchemy import Column, String, Float, Boolean, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Alert(Base):
    __tablename__ = 'alerts'

    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    alert_type = Column(Enum('price', 'signal', 'portfolio'))
    ticker = Column(String, nullable=True)
    condition = Column(String)  # JSON으로 조건 저장
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime)
    triggered_at = Column(DateTime, nullable=True)
```

**옵션 2: JSON 파일 (MVP)**
```python
# alerts.json
{
    "price_alerts": [...],
    "signal_alerts": [...],
    "portfolio_alerts": [...]
}
```

---

## 5. 구현 우선순위

### Phase 1: MVP (1주)

1. **브라우저 알림** - 가장 단순
   - Web Notification API 구현
   - Zustand alertStore 추가
   - 가격 알림 + 신호 알림

2. **WebSocket 통합**
   - `alert_triggered` 메시지 타입 추가
   - 클라이언트 핸들러 구현

### Phase 2: 확장 (2주)

3. **Telegram 연동**
   - Bot 생성 및 연동
   - 사용자 chat_id 저장 UI

4. **이메일 알림**
   - SMTP 설정
   - 템플릿 시스템

### Phase 3: 고급 기능 (3주)

5. **Slack 연동**
6. **포트폴리오 알림**
7. **알림 히스토리 대시보드**

---

## 6. API 엔드포인트 설계

### 6.1 알림 CRUD

```
# 알림 목록 조회
GET /api/alerts
Response: { alerts: [...] }

# 가격 알림 생성
POST /api/alerts/price
Body: { ticker, target_price, condition }
Response: { id, ... }

# 신호 알림 생성
POST /api/alerts/signal
Body: { ticker, signal_type, min_probability }
Response: { id, ... }

# 알림 삭제
DELETE /api/alerts/{alert_id}
Response: { success: true }

# 알림 활성화/비활성화
PATCH /api/alerts/{alert_id}
Body: { enabled: true/false }
Response: { ... }
```

### 6.2 채널 설정

```
# Telegram 연동
POST /api/alerts/channels/telegram
Body: { chat_id }

# 이메일 설정
POST /api/alerts/channels/email
Body: { email }

# 채널 설정 조회
GET /api/alerts/channels
Response: { telegram: {...}, email: {...}, ... }
```

---

## 7. WebSocket 메시지 확장

### 7.1 새 메시지 타입

```javascript
// 알림 발생 시 클라이언트로 전송
{
    "type": "alert_triggered",
    "alert_type": "price",  // price, signal, portfolio
    "alert_id": "abc123",
    "ticker": "AAPL",
    "data": {
        "target_price": 200.00,
        "current_price": 200.15,
        "condition": "above"
    },
    "timestamp": "2025-12-23T12:00:00Z"
}
```

### 7.2 websocket.py 수정 예시

```python
async def broadcast_alert(alert_type: str, alert_data: dict):
    """알림 발생 시 클라이언트에 broadcast"""
    message = {
        "type": "alert_triggered",
        "alert_type": alert_type,
        **alert_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(message)
```

---

## 8. 요약

### 8.1 권장 구현 순서

| 순위 | 항목 | 난이도 | 예상 기간 |
|------|------|--------|----------|
| 1 | 브라우저 알림 + 가격 알림 | 낮음 | 2일 |
| 2 | 신호 알림 | 낮음 | 1일 |
| 3 | Telegram 연동 | 중간 | 2일 |
| 4 | 이메일 알림 | 중간 | 2일 |
| 5 | 포트폴리오 알림 | 중간 | 2일 |
| 6 | Slack 연동 | 중간 | 1일 |

### 8.2 필요 패키지

```bash
# Backend
pip install python-telegram-bot
pip install aiosmtplib
pip install jinja2

# 선택적
pip install slack-sdk
```

### 8.3 환경변수 (settings.py 또는 .env)

```bash
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_app_password
FROM_EMAIL=alerts@fiveforfree.com

# Slack (선택)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

---

*분석팀장 작성*
