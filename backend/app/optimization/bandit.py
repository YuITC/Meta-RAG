import math
import random
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.config import RETRIEVAL_CONFIGS
from app.database import async_session_factory
from app.models.db_models import BanditArm
from app.memory.strategy_memory import StrategyMemory


class BanditOptimizer:
    """UCB1 multi-armed bandit for retrieval config selection."""

    def __init__(self):
        self._arms: dict[str, dict] = {
            config_id: {"count": 0, "total_reward": 0.0}
            for config_id in RETRIEVAL_CONFIGS
        }
        self._total_pulls: int = 0
        self._memory = StrategyMemory()

    async def initialize(self) -> None:
        """Load arm state from DB."""
        async with async_session_factory() as session:
            for config_id in RETRIEVAL_CONFIGS:
                result = await session.execute(
                    select(BanditArm).where(BanditArm.config_id == config_id)
                )
                arm = result.scalar_one_or_none()
                if arm:
                    self._arms[config_id] = {
                        "count": arm.count,
                        "total_reward": arm.total_reward,
                    }
                    self._total_pulls += arm.count
                else:
                    # Insert default row
                    session.add(BanditArm(config_id=config_id, count=0, total_reward=0.0))
            await session.commit()

    def _ucb1_score(self, config_id: str) -> float:
        arm = self._arms[config_id]
        if arm["count"] == 0:
            return float("inf")
        mean = arm["total_reward"] / arm["count"]
        exploration = math.sqrt(2 * math.log(max(self._total_pulls, 1)) / arm["count"])
        return mean + exploration

    async def select_config(self, query: str) -> str:
        """Select retrieval config using memory-aware UCB1."""
        # Try memory lookup first
        memory_config = await self._memory.get_best_config(query)
        if memory_config:
            return memory_config

        # UCB1 selection
        scores = {cid: self._ucb1_score(cid) for cid in RETRIEVAL_CONFIGS}
        best = max(scores, key=scores.get)
        return best

    def select_different_config(self, current_config_id: str) -> str:
        """Select a different config for retry (best UCB1 among others)."""
        others = [cid for cid in RETRIEVAL_CONFIGS if cid != current_config_id]
        if not others:
            return current_config_id
        scores = {cid: self._ucb1_score(cid) for cid in others}
        return max(scores, key=scores.get)

    async def update(self, config_id: str, reward: float) -> None:
        """Update arm with observed reward and persist to DB."""
        if config_id not in self._arms:
            return
        self._arms[config_id]["count"] += 1
        self._arms[config_id]["total_reward"] += reward
        self._total_pulls += 1

        async with async_session_factory() as session:
            result = await session.execute(
                select(BanditArm).where(BanditArm.config_id == config_id)
            )
            arm = result.scalar_one_or_none()
            if arm:
                arm.count = self._arms[config_id]["count"]
                arm.total_reward = self._arms[config_id]["total_reward"]
                arm.updated_at = datetime.now(timezone.utc)
            else:
                session.add(
                    BanditArm(
                        config_id=config_id,
                        count=self._arms[config_id]["count"],
                        total_reward=self._arms[config_id]["total_reward"],
                    )
                )
            await session.commit()

    def get_arm_stats(self) -> dict:
        stats = {}
        for cid, arm in self._arms.items():
            mean = arm["total_reward"] / arm["count"] if arm["count"] > 0 else 0.0
            stats[cid] = {
                "count": arm["count"],
                "mean_reward": round(mean, 4),
                "ucb1_score": round(self._ucb1_score(cid), 4),
                **RETRIEVAL_CONFIGS[cid],
            }
        return stats
