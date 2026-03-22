export default function AppLoading() {
  return (
    <div className="min-h-screen py-8 sm:py-12 px-4 sm:px-6">
      <div className="max-w-[900px] mx-auto space-y-6">
        {/* Header skeleton */}
        <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-6 animate-pulse">
          <div className="flex items-center gap-4">
            <div className="h-16 w-16 rounded-full bg-white/[0.06]" />
            <div className="flex-1 space-y-2">
              <div className="h-5 w-32 rounded bg-white/[0.06]" />
              <div className="h-3 w-24 rounded bg-white/[0.04]" />
            </div>
          </div>
        </div>
        {/* Content skeleton */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-6 animate-pulse h-40" />
          <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-6 animate-pulse h-40" />
        </div>
        <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-6 animate-pulse h-48" />
      </div>
    </div>
  );
}
