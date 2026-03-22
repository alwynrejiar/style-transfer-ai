"use client";

import { mockActivity } from "@/lib/mockData";

export function RecentActivity() {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-6">
      <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-white/30 mb-4">
        Recent Activity
      </p>
      <div className="space-y-2">
        {mockActivity.map((item) => (
          <div
            key={item.id}
            className="flex items-center justify-between rounded-xl border border-white/8 bg-white/[0.02] px-4 py-3"
          >
            <p className="text-sm font-medium text-white">{item.text}</p>
            <span className="text-xs text-white/30 shrink-0 ml-4">{item.time}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
