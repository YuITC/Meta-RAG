from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import BanditState, RunLog
from app.optimization.bandit import ThompsonSamplingBandit


async def load_bandit(db: AsyncSession, query_type: str) -> ThompsonSamplingBandit:
    """Load persisted bandit state from DB; returns fresh bandit if not found."""
    result = await db.execute(
        select(BanditState).where(BanditState.query_type == query_type)
    )
    state = result.scalar_one_or_none()
    bandit = ThompsonSamplingBandit()
    if state:
        bandit.alpha = dict(state.alpha)
        bandit.beta = dict(state.beta)
    return bandit


async def save_bandit(
    db: AsyncSession, query_type: str, bandit: ThompsonSamplingBandit
) -> None:
    """Upsert bandit state to DB."""
    result = await db.execute(
        select(BanditState).where(BanditState.query_type == query_type)
    )
    state = result.scalar_one_or_none()
    if state:
        state.alpha = dict(bandit.alpha)
        state.beta = dict(bandit.beta)
        state.updated_at = datetime.utcnow()
    else:
        state = BanditState(
            query_type=query_type,
            alpha=dict(bandit.alpha),
            beta=dict(bandit.beta),
        )
        db.add(state)
    await db.commit()


async def log_run(
    db: AsyncSession,
    query: str,
    query_type: str,
    config_name: str,
    faithfulness: float,
    cost: float,
    latency: float,
    utility: float,
    is_retry: bool = False,
) -> None:
    """Append a run record to the run log table."""
    entry = RunLog(
        query=query,
        query_type=query_type,
        config_name=config_name,
        faithfulness=faithfulness,
        cost=cost,
        latency=latency,
        utility=utility,
        is_retry=is_retry,
    )
    db.add(entry)
    await db.commit()
