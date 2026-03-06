import asyncio
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import select

from app.database import async_session_factory
from app.models.db_models import StrategyMemory as StrategyMemoryModel

MIN_SAMPLES_FOR_CLUSTERING = 10
N_CLUSTERS = 5


class StrategyMemory:
    """
    Stores (query_embedding, config_id, utility) history.
    Uses KMeans to learn which retrieval strategy works best per query cluster.
    """

    def __init__(self):
        self._kmeans: KMeans | None = None
        self._cluster_best: dict[int, str] = {}  # cluster_id -> best config_id

    async def store(
        self,
        query: str,
        config_id: str,
        utility: float,
        query_type: str | None = None,
    ) -> None:
        from app.retrieval.dense import embed_text
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, embed_text, query)

        async with async_session_factory() as session:
            session.add(
                StrategyMemoryModel(
                    query_embedding=embedding,
                    config_id=config_id,
                    utility=utility,
                    query_type=query_type,
                )
            )
            await session.commit()

        # Trigger re-clustering if enough samples
        await self._maybe_recluster()

    async def get_best_config(self, query: str) -> str | None:
        """Return best config for similar past queries, or None if not enough data."""
        if self._kmeans is None:
            await self._maybe_recluster()
        if self._kmeans is None:
            return None

        from app.retrieval.dense import embed_text
        import asyncio
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, embed_text, query)
        vec = np.array(embedding).reshape(1, -1)
        cluster_id = int(self._kmeans.predict(vec)[0])
        return self._cluster_best.get(cluster_id)

    async def _maybe_recluster(self) -> None:
        async with async_session_factory() as session:
            result = await session.execute(select(StrategyMemoryModel))
            records = result.scalars().all()

        if len(records) < MIN_SAMPLES_FOR_CLUSTERING:
            return

        embeddings = np.array([r.query_embedding for r in records])
        n_clusters = min(N_CLUSTERS, len(records))

        loop = asyncio.get_event_loop()
        kmeans = await loop.run_in_executor(
            None,
            lambda: KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(embeddings),
        )
        self._kmeans = kmeans

        # For each cluster, find the config with highest mean utility
        cluster_rewards: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for i, record in enumerate(records):
            cluster_id = int(kmeans.labels_[i])
            cluster_rewards[cluster_id][record.config_id].append(record.utility)

        best: dict[int, str] = {}
        for cluster_id, configs in cluster_rewards.items():
            best_config = max(configs, key=lambda c: np.mean(configs[c]))
            best[cluster_id] = best_config
        self._cluster_best = best
