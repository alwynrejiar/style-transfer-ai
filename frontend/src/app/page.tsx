import Link from "next/link";
import Image from "next/image";
import {
  Search, Brain, RefreshCw, BarChart3, Shield, Zap,
  ArrowRight,
} from "lucide-react";
import { MeshBackground } from "@/components/background/MeshBackground";
import { Button } from "@/components/ui/button";

const features = [
  {
    icon: Search,
    title: "7-Pass Style Analysis",
    description: "Lexical, syntactic, voice, discourse, rhythm, psycholinguistic, synthesis.",
  },
  {
    icon: Brain,
    title: "Cognitive Bridging",
    description: "Auto-inject contextual analogies for dense passages.",
  },
  {
    icon: RefreshCw,
    title: "Style Transfer",
    description: "Rewrite any content in a target author's voice.",
  },
  {
    icon: BarChart3,
    title: "Stylometric Profiles",
    description: "Save, load, and compare writing fingerprints.",
  },
  {
    icon: Shield,
    title: "Privacy-First",
    description: "Local Ollama processing, zero data leakage.",
  },
  {
    icon: Zap,
    title: "Multi-Model",
    description: "Ollama local, Remote Ollama, Google Gemini.",
  },
];

const steps = [
  { num: "01", label: "Input Text" },
  { num: "02", label: "Stylometric Analysis" },
  { num: "03", label: "Style Encoding" },
  { num: "04", label: "Neural Transformation" },
  { num: "05", label: "Output & Metrics" },
];

const marqueeTerms = [
  "Stylometric Analysis",
  "Style Encoding",
  "Neural Transformation",
  "Cognitive Bridging",
  "Keep Your Voice",
  "Rewrite Smarter",
];

export default function LandingPage() {
  return (
    <div className="relative min-h-screen bg-[#0a0a0a]">
      {/* Hero Section */}
      <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
        <MeshBackground
          colors={["#000000", "#0d0d1a", "#1a0a2e", "#0a0a0a"]}
          speed={0.6}
          className="absolute inset-0 w-full h-full"
        />

        {/* Nav */}
        <nav className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-6 sm:px-12 py-5">
          <div className="flex items-center gap-2.5">
            <Image src="/logo.png" alt="Stylomex" width={28} height={28} className="rounded-md" />
            <span className="font-semibold text-white text-sm tracking-tight">Stylomex.AI</span>
          </div>
          <Link href="/login">
            <Button variant="ghost" size="sm">
              Sign in <ArrowRight className="h-3.5 w-3.5 ml-1" />
            </Button>
          </Link>
        </nav>

        {/* Hero Content */}
        <div className="relative z-10 text-center px-6 max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 mb-8">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
            <span className="text-xs text-white/60 font-medium">✦ Advanced Stylometry AI</span>
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-7xl font-bold text-white tracking-tight leading-[1.1] mb-6">
            Understand & Transfer
            <br />
            Writing Styles
          </h1>

          <p className="text-base sm:text-lg text-white/50 max-w-xl mx-auto mb-8 leading-relaxed">
            Analyze any author&apos;s linguistic fingerprint. Generate content in their voice.
            Compare writing identities with forensic precision.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <Link href="/login">
              <Button size="lg" className="px-8">
                Start Analyzing <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </Link>
            <Link href="#features">
              <Button variant="outline" size="lg" className="px-8">
                View Demo
              </Button>
            </Link>
          </div>
        </div>

        {/* Marquee */}
        <div className="absolute bottom-0 left-0 right-0 z-10 border-t border-white/[0.04] bg-black/20 backdrop-blur-sm overflow-hidden py-4">
          <div className="flex animate-marquee whitespace-nowrap">
            {[...marqueeTerms, ...marqueeTerms, ...marqueeTerms, ...marqueeTerms].map((term, i) => (
              <span key={i} className="mx-6 text-sm text-white/20 font-medium flex items-center gap-3">
                <span className="h-1 w-1 rounded-full bg-white/10" />
                {term}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              Powerful Features
            </h2>
            <p className="text-white/40 max-w-lg mx-auto">
              Everything you need for advanced stylometry analysis and style transfer.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature, i) => (
              <div
                key={i}
                className="group rounded-2xl border border-white/8 bg-white/[0.02] p-6 transition-all duration-300 hover:bg-white/[0.04] hover:border-white/12 hover:-translate-y-1 hover:shadow-[0_8px_30px_rgba(108,92,231,0.08)]"
              >
                <div className="h-10 w-10 rounded-xl bg-[var(--accent)]/10 flex items-center justify-center mb-4 group-hover:bg-[var(--accent)]/20 transition-colors">
                  <feature.icon className="h-5 w-5 text-[var(--accent)]" />
                </div>
                <h3 className="font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-sm text-white/40 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="relative py-24 px-6 border-t border-white/[0.04]">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              How It Works
            </h2>
            <p className="text-white/40 max-w-lg mx-auto">
              Five steps from raw text to stylistic transformation.
            </p>
          </div>

          <div className="flex flex-col md:flex-row items-center justify-between gap-6 md:gap-0">
            {steps.map((step, i) => (
              <div key={i} className="flex items-center">
                <div className="flex flex-col items-center text-center">
                  <div className="h-14 w-14 rounded-full border border-white/10 bg-white/[0.04] flex items-center justify-center mb-3 text-lg font-bold text-[var(--accent)]">
                    {step.num}
                  </div>
                  <p className="text-sm text-white/60 font-medium max-w-[120px]">{step.label}</p>
                </div>
                {i < steps.length - 1 && (
                  <div className="hidden md:block w-12 lg:w-20 h-px bg-gradient-to-r from-white/10 to-white/5 mx-2" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[0.04] py-12 px-6">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2.5">
            <Image src="/logo.png" alt="Stylomex" width={24} height={24} className="rounded-md" />
            <span className="font-semibold text-white text-sm">Stylomex.AI</span>
            <span className="text-white/20 text-xs ml-2">Advanced Stylometry AI</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-white/40">
            <Link href="#features" className="hover:text-white/70 transition-colors">Features</Link>
            <Link href="#" className="hover:text-white/70 transition-colors">Demo</Link>
            <a href="https://github.com/alwynrejiar/style-transfer-ai" target="_blank" rel="noreferrer" className="hover:text-white/70 transition-colors">GitHub</a>
            <Link href="/contact" className="hover:text-white/70 transition-colors">Contact</Link>
          </div>
          <p className="text-xs text-white/20">© 2025 Stylomex.AI</p>
        </div>
      </footer>
    </div>
  );
}
