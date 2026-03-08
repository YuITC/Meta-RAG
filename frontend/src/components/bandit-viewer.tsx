"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";
import type { BanditStats } from "@/lib/types";

function ConfigBar({
  name,
  winRate,
  trials,
}: {
  name: string;
  winRate: number;
  trials: number;
}) {
  const pct = (winRate * 100).toFixed(1);
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">Config {name}</span>
        <span className="font-mono text-muted-foreground">
          {pct}% <span className="text-[10px]">({trials} trials)</span>
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-gray-100">
        <div
          className="h-1.5 rounded-full bg-blue-500 transition-all duration-500"
          style={{ width: `${Math.max(winRate * 100, 1)}%` }}
        />
      </div>
    </div>
  );
}

function BanditGroup({ stat }: { stat: BanditStats }) {
  return (
    <div className="space-y-2">
      <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        {stat.query_type.replace(/_/g, " ")}
      </p>
      {Object.entries(stat.configs).map(([name, cs]) => (
        <ConfigBar
          key={name}
          name={name}
          winRate={cs.win_rate}
          trials={cs.trials}
        />
      ))}
    </div>
  );
}

export function BanditViewer() {
  const banditStats = useStore((s) => s.banditStats);
  const fetchBanditStats = useStore((s) => s.fetchBanditStats);

  useEffect(() => {
    fetchBanditStats();
  }, [fetchBanditStats]);

  if (banditStats.length === 0) {
    return (
      <section className="rounded-lg border bg-background p-3">
        <div className="mb-3 border-b pb-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Strategy Learning
          </h3>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Thompson sampling win-rate estimates for each retrieval configuration.
          </p>
        </div>
        <div className="text-xs text-muted-foreground">
          No bandit data yet. Run queries to see strategy learning.
        </div>
      </section>
    );
  }

  return (
    <section className="rounded-lg border bg-background p-3">
      <div className="mb-3 border-b pb-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Strategy Learning
        </h3>
        <p className="mt-1 text-[11px] text-muted-foreground">
          Thompson sampling win-rate estimates for each retrieval configuration.
        </p>
      </div>

      <div className="space-y-4">
      {banditStats.map((s) => (
        <BanditGroup key={s.query_type} stat={s} />
      ))}
      </div>
    </section>
  );
}
