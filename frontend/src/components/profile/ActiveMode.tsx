"use client";

import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/useAppStore";

export function ActiveMode() {
  const { activeMode, modes } = useAppStore();
  const mode = modes.find((m) => m.name === activeMode);

  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-6">
      <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-white/30 mb-3">
        Active Mode
      </p>
      <h3 className="text-2xl font-bold text-white">{activeMode}</h3>
      <p className="text-sm text-white/40 mt-1">{mode?.desc || "No description"}</p>
      <Button variant="outline" size="sm" className="mt-4">
        Change Mode
      </Button>
    </div>
  );
}
