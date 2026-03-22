import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string | number;
  suffix?: string;
  trend?: "up" | "down" | "neutral";
}

export function StatCard({ label, value, suffix, trend }: StatCardProps) {
  return (
    <div className="rounded-xl border border-white/8 bg-white/[0.02] p-4">
      <p className="text-xs font-medium uppercase tracking-widest text-white/30 mb-2">
        {label}
      </p>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold font-mono text-white">
          {typeof value === "number" ? value.toLocaleString() : value}
        </span>
        {suffix && <span className="text-sm text-white/40">{suffix}</span>}
        {trend && trend !== "neutral" && (
          <span className={cn("text-xs ml-2", trend === "up" ? "text-green-400" : "text-red-400")}>
            {trend === "up" ? "↑" : "↓"}
          </span>
        )}
      </div>
    </div>
  );
}
