"use client";

import { StatCard } from "@/components/ui/stat-card";
import { mockStats } from "@/lib/mockData";

export function UsageSummary() {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-6">
      <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-white/30 mb-3">
        Usage Summary
      </p>
      <div className="grid grid-cols-1 gap-3">
        <StatCard label="Words Processed" value={mockStats.wordsProcessed} />
        <StatCard label="Humanized" value={mockStats.humanized} />
      </div>
    </div>
  );
}
