import type { Metadata } from "next";
import { LoginForm } from "@/components/auth/LoginForm";
import { NoiseOverlay } from "@/components/background/NoiseOverlay";

export const metadata: Metadata = {
  title: "Login | Stylomex.AI",
};

export default function LoginPage() {
  return (
    <div className="relative min-h-screen flex items-center justify-center bg-[#0a0a0a]">
      <NoiseOverlay opacity={0.02} />
      <div className="relative z-10 w-full px-4">
        <LoginForm />
      </div>
    </div>
  );
}
