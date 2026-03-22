"use client";

import { useEffect, useState } from "react";

interface MeshBackgroundProps {
  colors?: string[];
  speed?: number;
  className?: string;
}

export function MeshBackground({
  colors = ["#000000", "#0d0d1a", "#1a0a2e", "#0a0a0a"],
  speed = 0.6,
  className = "",
}: MeshBackgroundProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className={`bg-[#0a0a0a] ${className}`}>
        <div className="noise-overlay absolute inset-0" />
      </div>
    );
  }

  return (
    <div className={`relative overflow-hidden ${className}`}>
      {/* Animated gradient */}
      <div
        className="absolute inset-0 animate-gradient"
        style={{
          background: `
            radial-gradient(ellipse at 20% 50%, ${colors[2]}88 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, ${colors[1]}66 0%, transparent 50%),
            radial-gradient(ellipse at 40% 80%, ${colors[2]}44 0%, transparent 50%),
            radial-gradient(ellipse at 60% 40%, ${colors[1]}55 0%, transparent 60%),
            ${colors[3]}
          `,
          animation: `meshMove ${20 / speed}s ease-in-out infinite alternate`,
        }}
      />
      {/* Floating orbs */}
      <div
        className="absolute w-[600px] h-[600px] rounded-full opacity-20 blur-[120px]"
        style={{
          background: colors[2],
          top: "10%",
          left: "20%",
          animation: `floatOrb1 ${15 / speed}s ease-in-out infinite alternate`,
        }}
      />
      <div
        className="absolute w-[400px] h-[400px] rounded-full opacity-15 blur-[100px]"
        style={{
          background: colors[1],
          bottom: "20%",
          right: "10%",
          animation: `floatOrb2 ${18 / speed}s ease-in-out infinite alternate`,
        }}
      />
      {/* Noise overlay */}
      <div className="noise-overlay absolute inset-0" />
      <style jsx>{`
        @keyframes meshMove {
          0% { transform: scale(1) rotate(0deg); }
          100% { transform: scale(1.1) rotate(3deg); }
        }
        @keyframes floatOrb1 {
          0% { transform: translate(0, 0); }
          100% { transform: translate(80px, 60px); }
        }
        @keyframes floatOrb2 {
          0% { transform: translate(0, 0); }
          100% { transform: translate(-60px, -40px); }
        }
      `}</style>
    </div>
  );
}
