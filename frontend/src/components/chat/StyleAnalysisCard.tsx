"use client";

import { motion } from "framer-motion";
import {
  Brain,
  BookOpen,
  Target,
  Shield,
  AlertTriangle,
  CheckCircle2,
  Gauge,
} from "lucide-react";
import type { StyleAnalysisResult } from "@/lib/analyzeStyle";
import { cn } from "@/lib/utils";

interface StyleAnalysisCardProps {
  result: StyleAnalysisResult;
  saved: boolean;
}

const formalityColor: Record<string, string> = {
  casual: "bg-blue-500/20 text-blue-300 border-blue-500/30",
  "semi-formal": "bg-purple-500/20 text-purple-300 border-purple-500/30",
  formal: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  academic: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
};

const readabilityLabel = (score: number) => {
  if (score >= 80) return { label: "Very Easy", color: "text-emerald-400" };
  if (score >= 60) return { label: "Standard", color: "text-blue-400" };
  if (score >= 40) return { label: "Fairly Difficult", color: "text-amber-400" };
  return { label: "Difficult", color: "text-red-400" };
};

export function StyleAnalysisCard({ result, saved }: StyleAnalysisCardProps) {
  const flesch = result.readability_metrics?.flesch_reading_ease ?? 0;
  const readability = readabilityLabel(flesch);
  const formality = result.consolidated_analysis?.voice?.formality_level ?? "casual";
  const formalityClass = formalityColor[formality] ?? formalityColor["casual"];
  const barWidth = Math.max(4, Math.min(100, flesch));

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-full rounded-2xl border border-white/10 bg-white/[0.04] overflow-hidden"
    >
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/8 flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-lg bg-[var(--accent)]/20 flex items-center justify-center shrink-0">
            <Brain className="h-4 w-4 text-[var(--accent)]" />
          </div>
          <div>
            <p className="text-xs font-medium uppercase tracking-widest text-white/30">
              Style Analysis
            </p>
            <p className="text-sm text-white/60 mt-0.5">
              {result.word_count} words · {result.char_count} characters
            </p>
          </div>
        </div>
        {saved && (
          <span className="flex items-center gap-1 text-xs text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 rounded-full px-2.5 py-1 shrink-0">
            <CheckCircle2 className="h-3 w-3" />
            Saved to profile
          </span>
        )}
      </div>

      {/* Fingerprint */}
      <div className="px-5 py-4 border-b border-white/8">
        <p className="text-[10px] uppercase tracking-widest text-white/30 mb-1.5">
          Style Fingerprint
        </p>
        <p className="text-sm text-white leading-relaxed">
          {result.style_fingerprint_summary}
        </p>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-3 divide-x divide-white/8 border-b border-white/8">
        {/* Readability */}
        <div className="px-4 py-3">
          <div className="flex items-center gap-1.5 mb-1">
            <Gauge className="h-3.5 w-3.5 text-white/30" />
            <p className="text-[10px] uppercase tracking-widest text-white/30">Readability</p>
          </div>
          <p className={cn("text-sm font-semibold", readability.color)}>
            {flesch.toFixed(0)}
          </p>
          <p className="text-xs text-white/40">{readability.label}</p>
          {/* Bar */}
          <div className="mt-2 h-1 w-full rounded-full bg-white/10">
            <div
              className="h-1 rounded-full bg-[var(--accent)]"
              style={{ width: `${barWidth}%` }}
            />
          </div>
        </div>

        {/* Grade */}
        <div className="px-4 py-3">
          <div className="flex items-center gap-1.5 mb-1">
            <BookOpen className="h-3.5 w-3.5 text-white/30" />
            <p className="text-[10px] uppercase tracking-widest text-white/30">Grade Level</p>
          </div>
          <p className="text-sm font-semibold text-white">
            {result.readability_metrics?.estimated_reading_level ?? "—"}
          </p>
          <p className="text-xs text-white/40">
            FK {result.readability_metrics?.flesch_kincaid_grade?.toFixed(1) ?? "—"}
          </p>
        </div>

        {/* Formality */}
        <div className="px-4 py-3">
          <div className="flex items-center gap-1.5 mb-1">
            <Target className="h-3.5 w-3.5 text-white/30" />
            <p className="text-[10px] uppercase tracking-widest text-white/30">Formality</p>
          </div>
          <span
            className={cn(
              "inline-block text-xs font-medium px-2 py-0.5 rounded-full border capitalize",
              formalityClass
            )}
          >
            {formality}
          </span>
          <p className="text-xs text-white/40 mt-1 capitalize">
            {result.consolidated_analysis?.voice?.tone ?? "—"}
          </p>
        </div>
      </div>

      {/* Traits */}
      <div className="px-5 py-4 border-b border-white/8">
        <p className="text-[10px] uppercase tracking-widest text-white/30 mb-2.5">
          Distinctive Traits
        </p>
        <div className="flex flex-wrap gap-1.5">
          {result.most_distinctive_traits?.map((trait, i) => (
            <span
              key={i}
              className="text-xs px-2.5 py-1 rounded-full bg-[var(--accent)]/10 border border-[var(--accent)]/20 text-[var(--accent)]"
            >
              {trait}
            </span>
          ))}
        </div>
      </div>

      {/* Strengths + Weaknesses */}
      <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-white/8">
        {/* Strengths */}
        <div className="px-5 py-4">
          <div className="flex items-center gap-1.5 mb-2.5">
            <Shield className="h-3.5 w-3.5 text-emerald-400" />
            <p className="text-[10px] uppercase tracking-widest text-white/30">Keep These</p>
          </div>
          <ul className="space-y-1.5">
            {result.do_not_lose?.map((item, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                <span className="text-xs text-white/60">{item}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Weaknesses */}
        <div className="px-5 py-4">
          <div className="flex items-center gap-1.5 mb-2.5">
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
            <p className="text-[10px] uppercase tracking-widest text-white/30">Avoid</p>
          </div>
          <ul className="space-y-1.5">
            {result.avoid_in_rewrite?.map((item, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-400 shrink-0" />
                <span className="text-xs text-white/60">{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </motion.div>
  );
}
