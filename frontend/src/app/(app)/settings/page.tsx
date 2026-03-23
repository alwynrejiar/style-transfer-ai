"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Moon, Sun, Monitor, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { useAppStore } from "@/store/useAppStore";
import { useAuth } from "@/hooks/useAuth";
import { cn } from "@/lib/utils";

const modelOptions = ["gemma3:1b", "remote-ollama", "gemini"];
const domainOptions = [
  "Sports", "Gaming", "Cooking", "Nature", "Daily Life", "Tech", "General Simplification",
];

export default function SettingsPage() {
  const { modelConfig, updateModelConfig, analogyConfig, updateAnalogyConfig, theme, setTheme } = useAppStore();
  const { deleteAccount } = useAuth();
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleDelete = async () => {
    setDeleting(true);
    setDeleteError(null);
    const { error } = await deleteAccount();
    if (error) {
      setDeleteError(error);
      setDeleting(false);
    } else {
      setDeleteOpen(false);
    }
  };

  const themes = [
    { value: "dark" as const, icon: Moon, label: "Dark" },
    { value: "light" as const, icon: Sun, label: "Light" },
    { value: "system" as const, icon: Monitor, label: "System" },
  ];

  return (
    <div className="min-h-screen py-8 sm:py-12 px-4 sm:px-6">
      <div className="max-w-[900px] mx-auto space-y-6">
        {/* Model Configuration */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="rounded-2xl border border-white/8 bg-white/[0.04] p-6"
        >
          <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-white/30 mb-4">
            Model Configuration
          </p>

          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-white">Use Local (Ollama)</Label>
                <p className="text-xs text-white/30 mt-0.5">Process all text locally for maximum privacy</p>
              </div>
              <Switch
                checked={modelConfig.useLocal}
                onCheckedChange={(checked) => updateModelConfig({ useLocal: checked })}
              />
            </div>

            <div className="space-y-2">
              <Label>Model Name</Label>
              <select
                value={modelConfig.modelName}
                onChange={(e) => updateModelConfig({ modelName: e.target.value })}
                className="flex h-11 w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-[var(--accent)] cursor-pointer"
              >
                {modelOptions.map((opt) => (
                  <option key={opt} value={opt} className="bg-[#111] text-white">
                    {opt}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <Label>Google Gemini API Key</Label>
              <Input
                type="password"
                placeholder="Enter your Google Gemini API key"
                value={modelConfig.geminiKey}
                onChange={(e) => updateModelConfig({ geminiKey: e.target.value })}
              />
            </div>
          </div>
        </motion.div>

        {/* Analogy Engine */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.08 }}
          className="rounded-2xl border border-white/8 bg-white/[0.04] p-6"
        >
          <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-white/30 mb-4">
            Analogy Engine Defaults
          </p>

          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-white">Enable analogy augmentation by default</Label>
                <p className="text-xs text-white/30 mt-0.5">Auto-inject contextual analogies for dense passages</p>
              </div>
              <Switch
                checked={analogyConfig.enabled}
                onCheckedChange={(checked) => updateAnalogyConfig({ enabled: checked })}
              />
            </div>

            <div className="space-y-2">
              <Label>Default Domain</Label>
              <select
                value={analogyConfig.defaultDomain}
                onChange={(e) => updateAnalogyConfig({ defaultDomain: e.target.value })}
                className="flex h-11 w-full rounded-lg border border-white/12 bg-white/5 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-[var(--accent)] cursor-pointer"
              >
                {domainOptions.map((opt) => (
                  <option key={opt} value={opt} className="bg-[#111] text-white">
                    {opt}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </motion.div>

        {/* Appearance */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.16 }}
          className="rounded-2xl border border-white/8 bg-white/[0.04] p-6"
        >
          <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-white/30 mb-4">
            Appearance
          </p>

          <div className="grid grid-cols-3 gap-3">
            {themes.map((t) => (
              <button
                key={t.value}
                onClick={() => setTheme(t.value)}
                className={cn(
                  "flex flex-col items-center gap-2 p-4 rounded-xl border transition-all duration-200 cursor-pointer",
                  theme === t.value
                    ? "border-[var(--accent)] bg-[var(--accent)]/5 text-white"
                    : "border-white/8 bg-white/[0.02] text-white/40 hover:border-white/16"
                )}
              >
                <t.icon className="h-5 w-5" />
                <span className="text-xs font-medium">{t.label}</span>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Danger Zone */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.24 }}
          className="rounded-2xl border border-[var(--destructive)]/20 bg-[var(--destructive)]/[0.02] p-6"
        >
          <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-[var(--destructive)]/60 mb-4">
            Danger Zone
          </p>

          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-white">Delete your account</p>
              <p className="text-xs text-white/30 mt-0.5">This action cannot be undone and deletes all your data</p>
            </div>
            <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
              <DialogTrigger asChild>
                <Button variant="destructive" size="sm">
                  <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                  Delete Account
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Are you absolutely sure?</DialogTitle>
                  <DialogDescription>
                    This will permanently delete your account, your style analyses, generated content, and remove your data from our servers.
                  </DialogDescription>
                  {deleteError && (
                    <div className="mt-2 p-3 rounded-lg bg-[var(--destructive)]/20 border border-[var(--destructive)]/50 text-[var(--destructive)] text-sm">
                      {deleteError}
                    </div>
                  )}
                </DialogHeader>
                <DialogFooter>
                  <Button variant="ghost" onClick={() => setDeleteOpen(false)} disabled={deleting}>Cancel</Button>
                  <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
                    {deleting ? "Deleting..." : "Delete Account"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
