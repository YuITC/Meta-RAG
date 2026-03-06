"""Unit tests for metric computation and bandit optimizer."""
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.optimization.bandit import BanditOptimizer
from app.config import RETRIEVAL_CONFIGS


class TestBanditOptimizer:
    def _make_bandit(self):
        bandit = BanditOptimizer()
        bandit._memory.get_best_config = AsyncMock(return_value=None)
        return bandit

    def test_ucb1_score_infinite_for_unplayed_arm(self):
        bandit = self._make_bandit()
        score = bandit._ucb1_score("A")
        assert score == float("inf")

    def test_ucb1_score_finite_after_update(self):
        bandit = self._make_bandit()
        bandit._arms["A"]["count"] = 3
        bandit._arms["A"]["total_reward"] = 2.1
        bandit._total_pulls = 5
        score = bandit._ucb1_score("A")
        expected_mean = 2.1 / 3
        expected_exploration = math.sqrt(2 * math.log(5) / 3)
        assert abs(score - (expected_mean + expected_exploration)) < 1e-9

    @pytest.mark.asyncio
    async def test_select_config_returns_valid_config(self):
        bandit = self._make_bandit()
        config_id = await bandit.select_config("test query")
        assert config_id in RETRIEVAL_CONFIGS

    def test_select_different_config(self):
        bandit = self._make_bandit()
        bandit._arms["A"]["count"] = 5
        bandit._arms["A"]["total_reward"] = 2.0
        bandit._arms["B"]["count"] = 3
        bandit._arms["B"]["total_reward"] = 1.5
        bandit._arms["C"]["count"] = 2
        bandit._arms["C"]["total_reward"] = 1.0
        bandit._total_pulls = 10

        alt = bandit.select_different_config("A")
        assert alt != "A"
        assert alt in RETRIEVAL_CONFIGS

    @pytest.mark.asyncio
    async def test_update_increments_count(self):
        bandit = self._make_bandit()
        # Patch DB write
        with patch("app.optimization.bandit.async_session_factory") as mock_sf:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.execute = AsyncMock(return_value=AsyncMock(scalar_one_or_none=lambda: None))
            mock_session.add = MagicMock()
            mock_session.commit = AsyncMock()
            mock_sf.return_value = mock_session

            await bandit.update("A", 0.8)
            assert bandit._arms["A"]["count"] == 1
            assert abs(bandit._arms["A"]["total_reward"] - 0.8) < 1e-9
            assert bandit._total_pulls == 1

    def test_get_arm_stats_has_all_configs(self):
        bandit = self._make_bandit()
        stats = bandit.get_arm_stats()
        assert set(stats.keys()) == set(RETRIEVAL_CONFIGS.keys())
        for cid, info in stats.items():
            assert "count" in info
            assert "mean_reward" in info
            assert "top_k" in info
