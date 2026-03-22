export function NoiseOverlay({ opacity = 0.03 }: { opacity?: number }) {
  return (
    <div
      className="noise-overlay absolute inset-0 pointer-events-none z-10"
      style={{ opacity }}
    />
  );
}
