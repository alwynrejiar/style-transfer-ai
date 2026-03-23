"use client";

import { RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { useStyleFeatureCycle } from "@/hooks/useStyleFeatureCycle";

export function FeatureCycleButton() {
  const router = useRouter();
  const { hiddenFeature, rotateHiddenFeature } = useStyleFeatureCycle();

  const handleClick = () => {
    const href = hiddenFeature.href;
    rotateHiddenFeature();
    router.push(href);
  };

  return (
    <Button
      type="button"
      variant="ghost"
      onClick={handleClick}
      className="h-8 px-2.5 text-xs text-white/65 hover:text-white hover:bg-white/[0.08]"
    >
      <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
      <span className="hidden sm:inline">{hiddenFeature.label}</span>
      <span className="sm:hidden">Mode</span>
    </Button>
  );
}
