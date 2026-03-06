from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text
from app.database import Base


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    query_type = Column(String(50))
    config_id = Column(String(10))
    faithfulness = Column(Float)
    citation_grounding = Column(Float)
    utility = Column(Float)
    cost = Column(Float)
    latency = Column(Float)
    answer = Column(Text)
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class BanditArm(Base):
    __tablename__ = "bandit_arms"

    config_id = Column(String(10), primary_key=True)
    count = Column(Integer, default=0)
    total_reward = Column(Float, default=0.0)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class StrategyMemory(Base):
    __tablename__ = "strategy_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_embedding = Column(JSON, nullable=False)  # list[float]
    config_id = Column(String(10), nullable=False)
    utility = Column(Float, nullable=False)
    query_type = Column(String(50))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
