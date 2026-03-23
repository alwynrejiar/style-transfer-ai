"use client";

import { useEffect, useRef } from "react";
import { ChatMessage } from "./ChatMessage";
import { StyleAnalysisCard } from "./StyleAnalysisCard";
import type { Message } from "@/lib/mockData";

interface ChatThreadProps {
  messages: Message[];
}

export function ChatThread({ messages }: ChatThreadProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="max-w-[720px] mx-auto space-y-6">
        {messages.map((message) =>
          message.analysisResult ? (
            <div key={message.id} className="mr-auto w-full max-w-[85%]">
              <StyleAnalysisCard
                result={message.analysisResult}
                saved={message.content === "saved"}
              />
            </div>
          ) : (
            <ChatMessage
              key={message.id}
              role={message.role}
              content={message.content}
              timestamp={message.timestamp}
            />
          )
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
