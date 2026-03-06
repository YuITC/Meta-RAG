from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RunLog(Base):
    __tablename__ = "run_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String)
    query_type: Mapped[str] = mapped_column(String)
    config_name: Mapped[str] = mapped_column(String)
    faithfulness: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float)
    latency: Mapped[float] = mapped_column(Float)
    utility: Mapped[float] = mapped_column(Float)
    is_retry: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BanditState(Base):
    __tablename__ = "bandit_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_type: Mapped[str] = mapped_column(String, unique=True, index=True)
    alpha: Mapped[dict] = mapped_column(JSONB)
    beta: Mapped[dict] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
