"use client";

import Image from "next/image";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export function ChatMessage({ role, content, timestamp }: ChatMessageProps) {
  const isUser = role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 max-w-[85%]",
        isUser ? "ml-auto flex-row-reverse" : "mr-auto"
      )}
    >
      {!isUser && (
        <div className="shrink-0 mt-1">
          <Image
            src="/logo.png"
            alt="Stylomex"
            width={28}
            height={28}
            className="rounded-md"
          />
        </div>
      )}
      <div className={cn("flex flex-col", isUser ? "items-end" : "items-start")}>
        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap",
            isUser
              ? "bg-white/8 text-white"
              : "text-white/90"
          )}
        >
          {renderContent(content)}
        </div>
        <span className="text-[10px] text-white/20 mt-1 px-1">
          {timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
    </div>
  );
}

function renderContent(content: string) {
  // Parse basic markdown-like formatting
  const sections = content.split("\n\n");
  return sections.map((section, i) => {
    if (section.startsWith("**") && section.includes("**\n")) {
      // Bold header section
      const [header, ...rest] = section.split("\n");
      return (
        <div key={i} className="mb-3 last:mb-0">
          <p className="font-semibold text-white mb-1">{header.replace(/\*\*/g, "")}</p>
          {rest.map((line, j) => (
            <p key={j} className="text-white/70 text-sm">{line.replace(/^- /, "• ")}</p>
          ))}
        </div>
      );
    }
    if (section.startsWith("---")) {
      return <div key={i} className="border-t border-white/10 my-3" />;
    }
    return (
      <p key={i} className="mb-2 last:mb-0">
        {section.split("**").map((part, j) =>
          j % 2 === 1 ? (
            <strong key={j} className="font-semibold text-white">{part}</strong>
          ) : (
            <span key={j}>{part}</span>
          )
        )}
      </p>
    );
  });
}
