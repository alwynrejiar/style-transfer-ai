"use client";

import { Button } from "@/components/ui/button";
import { ModeCard } from "@/components/ui/mode-card";
import { useAppStore } from "@/store/useAppStore";

export function SavedModes() {
  const { modes, setActiveMode } = useAppStore();

  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-6">
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-white/30 mb-1">
            Saved Modes
          </p>
          <p className="text-sm text-white/40">Quickly switch between your presets.</p>
        </div>
        <Button variant="outline" size="sm">
          Create New Mode
        </Button>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {modes.map((mode) => (
          <ModeCard
            key={mode.id}
            name={mode.name}
            description={mode.desc}
            active={mode.active}
            onSelect={() => setActiveMode(mode.name)}
          />
        ))}
      </div>
    </div>
  );
}
