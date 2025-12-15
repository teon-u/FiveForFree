"""SQLite database management with SQLAlchemy 2.0."""

from datetime import datetime
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    String,
    Float,
    Integer,
    DateTime,
    Boolean,
    Index,
    JSON,
    BigInteger,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    Session,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

from config.settings import settings


# Declarative base
class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# Database models
class Ticker(Base):
    """Ticker information for tracked stocks."""

    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    added_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("idx_ticker_symbol_active", "symbol", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Ticker(symbol={self.symbol}, name={self.name})>"


class MinuteBar(Base):
    """1-minute OHLCV bar data."""

    __tablename__ = "minute_bars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    vwap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trade_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Additional computed fields
    spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ask: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("idx_minute_symbol_timestamp", "symbol", "timestamp", unique=True),
        Index("idx_minute_ticker_timestamp", "ticker_id", "timestamp"),
        Index("idx_minute_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<MinuteBar(symbol={self.symbol}, timestamp={self.timestamp}, close={self.close})>"


class Prediction(Base):
    """Model predictions for price movements."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Prediction metadata
    prediction_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )
    target_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False)

    # Current price
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)

    # Predictions
    prob_up: Mapped[float] = mapped_column(Float, nullable=False)
    prob_down: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_direction: Mapped[str] = mapped_column(String(10), nullable=False)

    # Target thresholds
    target_percent: Mapped[float] = mapped_column(Float, nullable=False)

    # Actual outcome (filled after target_time)
    actual_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_max_gain_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_max_loss_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hit_target: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Model metadata
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("idx_pred_symbol_time", "symbol", "prediction_time"),
        Index("idx_pred_model_time", "model_type", "prediction_time"),
        Index("idx_pred_target_time", "target_time"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(symbol={self.symbol}, model={self.model_type}, "
            f"prob_up={self.prob_up:.2f}, prob_down={self.prob_down:.2f})>"
        )


class Trade(Base):
    """Executed trades for backtesting and live trading."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    # Trade details
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Position
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # long/short
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    position_size: Mapped[float] = mapped_column(Float, nullable=False)

    # Results
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    commission: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    net_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Target tracking
    target_percent: Mapped[float] = mapped_column(Float, nullable=False)
    hit_target: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    max_gain_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_loss_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Associated prediction
    prediction_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Trade type
    is_backtest: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_live: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Metadata
    notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("idx_trade_symbol_entry", "symbol", "entry_time"),
        Index("idx_trade_backtest", "is_backtest", "entry_time"),
        Index("idx_trade_live", "is_live", "entry_time"),
    )

    def __repr__(self) -> str:
        return (
            f"<Trade(symbol={self.symbol}, direction={self.direction}, "
            f"entry={self.entry_price}, exit={self.exit_price}, pnl_pct={self.pnl_percent})>"
        )


class ModelPerformance(Base):
    """Model performance metrics over time."""

    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, index=True)

    # Time period
    evaluation_date: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Prediction metrics
    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)

    # Direction-specific metrics
    up_predictions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    down_predictions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    up_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    down_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Confidence metrics
    avg_prob_correct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_prob_incorrect: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Financial metrics
    total_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_pnl_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Model metadata
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    training_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feature_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Additional metrics
    metrics_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("idx_perf_model_date", "model_type", "evaluation_date"),
        Index("idx_perf_symbol_date", "symbol", "evaluation_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelPerformance(model={self.model_type}, symbol={self.symbol}, "
            f"accuracy={self.accuracy:.2f}, win_rate={self.win_rate})>"
        )


# Database engine and session management
engine = None
SessionLocal = None


def init_db(database_url: Optional[str] = None) -> None:
    """
    Initialize the database engine and create all tables.

    Args:
        database_url: Database URL. If None, uses settings.DATABASE_URL
    """
    global engine, SessionLocal

    url = database_url or settings.DATABASE_URL

    # Create engine with appropriate settings
    connect_args = {}
    poolclass = None

    if url.startswith("sqlite"):
        # SQLite-specific settings
        connect_args = {"check_same_thread": False}
        # Use StaticPool for in-memory databases
        if ":memory:" in url:
            poolclass = StaticPool

    engine = create_engine(
        url,
        connect_args=connect_args,
        poolclass=poolclass,
        echo=False,  # Set to True for SQL debugging
    )

    # Create session factory
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get a database session with automatic cleanup.

    Yields:
        Session: SQLAlchemy session

    Example:
        >>> with get_db() as db:
        ...     tickers = db.execute(select(Ticker)).scalars().all()
    """
    if SessionLocal is None:
        init_db()

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# Query utility functions
def get_ticker_by_symbol(db: Session, symbol: str) -> Optional[Ticker]:
    """
    Get ticker by symbol.

    Args:
        db: Database session
        symbol: Ticker symbol

    Returns:
        Ticker object or None if not found
    """
    stmt = select(Ticker).where(Ticker.symbol == symbol)
    return db.execute(stmt).scalar_one_or_none()


def get_or_create_ticker(db: Session, symbol: str, **kwargs) -> Ticker:
    """
    Get existing ticker or create new one.

    Args:
        db: Database session
        symbol: Ticker symbol
        **kwargs: Additional ticker attributes

    Returns:
        Ticker object
    """
    ticker = get_ticker_by_symbol(db, symbol)
    if ticker is None:
        ticker = Ticker(symbol=symbol, **kwargs)
        db.add(ticker)
        db.flush()
    return ticker


def get_minute_bars(
    db: Session,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
) -> list[MinuteBar]:
    """
    Get minute bars for a symbol within a time range.

    Args:
        db: Database session
        symbol: Ticker symbol
        start_time: Start timestamp
        end_time: End timestamp

    Returns:
        List of MinuteBar objects
    """
    stmt = (
        select(MinuteBar)
        .where(MinuteBar.symbol == symbol)
        .where(MinuteBar.timestamp >= start_time)
        .where(MinuteBar.timestamp <= end_time)
        .order_by(MinuteBar.timestamp)
    )
    return list(db.execute(stmt).scalars().all())


def get_recent_predictions(
    db: Session,
    symbol: Optional[str] = None,
    model_type: Optional[str] = None,
    limit: int = 100,
) -> list[Prediction]:
    """
    Get recent predictions with optional filtering.

    Args:
        db: Database session
        symbol: Filter by symbol (optional)
        model_type: Filter by model type (optional)
        limit: Maximum number of results

    Returns:
        List of Prediction objects
    """
    stmt = select(Prediction)

    if symbol:
        stmt = stmt.where(Prediction.symbol == symbol)
    if model_type:
        stmt = stmt.where(Prediction.model_type == model_type)

    stmt = stmt.order_by(Prediction.prediction_time.desc()).limit(limit)

    return list(db.execute(stmt).scalars().all())


def get_model_performance_summary(
    db: Session,
    model_type: str,
    days: int = 30,
) -> Optional[ModelPerformance]:
    """
    Get the most recent performance metrics for a model.

    Args:
        db: Database session
        model_type: Model type to query
        days: Look back this many days

    Returns:
        ModelPerformance object or None
    """
    cutoff_date = datetime.utcnow()
    from datetime import timedelta
    cutoff_date = cutoff_date - timedelta(days=days)

    stmt = (
        select(ModelPerformance)
        .where(ModelPerformance.model_type == model_type)
        .where(ModelPerformance.evaluation_date >= cutoff_date)
        .order_by(ModelPerformance.evaluation_date.desc())
        .limit(1)
    )

    return db.execute(stmt).scalar_one_or_none()
