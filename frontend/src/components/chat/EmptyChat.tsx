"use client";

import Image from "next/image";

export function EmptyChat() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center">
        <Image
          src="/logo.png"
          alt="Stylomex"
          width={64}
          height={64}
          className="mx-auto opacity-[0.06] mb-4"
        />
        <p className="text-white/20 text-sm">Start a conversation</p>
      </div>
    </div>
  );
}
