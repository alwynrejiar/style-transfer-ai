export default function ChatLoading() {
  return (
    <div className="relative flex flex-col h-screen bg-[#0a0a0a]">
      {/* Top bar */}
      <div className="flex items-center justify-center py-3 border-b border-white/[0.04]">
        <div className="h-4 w-24 rounded bg-white/[0.06] animate-pulse" />
      </div>
      {/* Content */}
      <div className="flex-1 flex items-center justify-center">
        <div className="animate-pulse space-y-3 w-full max-w-md px-4">
          <div className="h-12 rounded-xl bg-white/[0.04] w-3/4 ml-auto" />
          <div className="h-20 rounded-xl bg-white/[0.03] w-4/5" />
          <div className="h-12 rounded-xl bg-white/[0.04] w-2/3 ml-auto" />
        </div>
      </div>
      {/* Input placeholder */}
      <div className="p-4 pb-6 flex justify-center">
        <div className="w-full max-w-[640px] h-[72px] rounded-2xl border border-white/8 bg-white/[0.02] animate-pulse" />
      </div>
    </div>
  );
}
