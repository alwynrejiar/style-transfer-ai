"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type StyleFeatureKey = "transfer" | "compare" | "analyze" | "student-analogy";

export interface StyleFeatureItem {
  key: StyleFeatureKey;
  label: string;
  href: string;
}

export const STYLE_FEATURES: StyleFeatureItem[] = [
  { key: "transfer", label: "Style Transformation", href: "/chat?mode=transfer" },
  { key: "compare", label: "Style Comparison", href: "/chat?mode=compare" },
  { key: "analyze", label: "Style Analysis", href: "/chat?mode=analyze" },
  { key: "student-analogy", label: "Style Analogy", href: "/chat?mode=student-analogy" },
];

const STORAGE_KEY = "stylomex.hidden-style-feature-index";
const CYCLE_EVENT = "stylomex-style-feature-cycle-changed";

function normalizeIndex(index: number): number {
  const count = STYLE_FEATURES.length;
  return ((index % count) + count) % count;
}

function readHiddenFeatureIndex(): number {
  if (typeof window === "undefined") return 0;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  const parsed = raw ? Number.parseInt(raw, 10) : 0;
  return Number.isNaN(parsed) ? 0 : normalizeIndex(parsed);
}

function writeHiddenFeatureIndex(index: number): void {
  if (typeof window === "undefined") return;
  const normalized = normalizeIndex(index);
  window.localStorage.setItem(STORAGE_KEY, String(normalized));
  window.dispatchEvent(new Event(CYCLE_EVENT));
}

export function useStyleFeatureCycle() {
  const [hiddenFeatureIndex, setHiddenFeatureIndexState] = useState(0);

  useEffect(() => {
    setHiddenFeatureIndexState(readHiddenFeatureIndex());

    const onStorage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY) {
        setHiddenFeatureIndexState(readHiddenFeatureIndex());
      }
    };

    const onCycleUpdate = () => {
      setHiddenFeatureIndexState(readHiddenFeatureIndex());
    };

    window.addEventListener("storage", onStorage);
    window.addEventListener(CYCLE_EVENT, onCycleUpdate);

    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(CYCLE_EVENT, onCycleUpdate);
    };
  }, []);

  const setHiddenFeatureIndex = useCallback((nextIndex: number) => {
    writeHiddenFeatureIndex(nextIndex);
    setHiddenFeatureIndexState(readHiddenFeatureIndex());
  }, []);

  const rotateHiddenFeature = useCallback(() => {
    const nextIndex = normalizeIndex(hiddenFeatureIndex + 1);
    setHiddenFeatureIndex(nextIndex);
    return nextIndex;
  }, [hiddenFeatureIndex, setHiddenFeatureIndex]);

  const hiddenFeature = useMemo(
    () => STYLE_FEATURES[normalizeIndex(hiddenFeatureIndex)],
    [hiddenFeatureIndex]
  );

  const sidebarStyleFeatures = useMemo(
    () => STYLE_FEATURES.filter((_, index) => index !== normalizeIndex(hiddenFeatureIndex)),
    [hiddenFeatureIndex]
  );

  return {
    hiddenFeatureIndex,
    hiddenFeature,
    sidebarStyleFeatures,
    setHiddenFeatureIndex,
    rotateHiddenFeature,
  };
}
