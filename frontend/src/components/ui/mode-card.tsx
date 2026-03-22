"use client";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface ModeCardProps {
  name: string;
  description: string;
  active?: boolean;
  onSelect?: () => void;
}

export function ModeCard({ name, description, active, onSelect }: ModeCardProps) {
  return (
    <div
      className={cn(
        "rounded-xl border p-4 transition-all duration-200",
        active
          ? "border-[var(--accent)] bg-[var(--accent)]/5"
          : "border-white/8 bg-white/[0.02] hover:border-white/16 hover:bg-white/[0.04]"
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h4 className="font-semibold text-white text-sm">{name}</h4>
          <p className="text-xs text-white/40 mt-1 line-clamp-2">{description}</p>
        </div>
        {active ? (
          <span className="shrink-0 inline-flex items-center rounded-full bg-[var(--accent)]/20 px-2.5 py-0.5 text-xs font-medium text-[var(--accent-secondary)]">
            Active
          </span>
        ) : (
          <Button variant="outline" size="sm" onClick={onSelect} className="shrink-0">
            Select
          </Button>
        )}
      </div>
    </div>
  );
}
