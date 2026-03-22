"use client";

import { Button } from "@/components/ui/button";

export default function AuthError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0a0a0a] px-4">
      <div className="text-center max-w-md">
        <h2 className="text-xl font-bold text-white mb-2">Authentication Error</h2>
        <p className="text-sm text-white/40 mb-6">{error.message || "Something went wrong."}</p>
        <Button onClick={reset}>Try Again</Button>
      </div>
    </div>
  );
}
