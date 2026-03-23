"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { Smile, Paperclip, Mic, Send } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSend, disabled, placeholder }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);

  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [value, adjustHeight]);

  const handleSend = () => {
    if (!value.trim() || disabled) return;
    onSend(value.trim());
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div
      className={`w-full max-w-[640px] mx-auto rounded-2xl border transition-all duration-200 ${
        focused
          ? "border-[var(--accent)]/40 shadow-[0_0_20px_rgba(108,92,231,0.1)]"
          : "border-white/12"
      } bg-[rgba(20,20,20,0.9)] backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.5)]`}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        placeholder={placeholder || "Type your message here..."}
        disabled={disabled}
        rows={1}
        className="w-full resize-none bg-transparent px-4 pt-4 pb-2 text-sm text-white placeholder:text-white/30 focus:outline-none disabled:opacity-50"
      />
      <div className="flex items-center justify-between px-3 pb-3">
        <div className="flex items-center gap-1">
          <button className="p-2 rounded-lg text-white/30 hover:text-white/60 hover:bg-white/5 transition-colors cursor-pointer">
            <Smile className="h-4 w-4" />
          </button>
          <button className="p-2 rounded-lg text-white/30 hover:text-white/60 hover:bg-white/5 transition-colors cursor-pointer">
            <Paperclip className="h-4 w-4" />
          </button>
        </div>
        <div className="flex items-center gap-2">
          <button className="p-2 rounded-full bg-white/10 text-white/40 hover:bg-white/20 transition-colors cursor-pointer">
            <Mic className="h-4 w-4" />
          </button>
          <button
            onClick={handleSend}
            disabled={!value.trim() || disabled}
            className="p-2 rounded-full bg-white text-black hover:bg-white/90 disabled:opacity-30 disabled:cursor-not-allowed transition-all active:scale-95 cursor-pointer"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
