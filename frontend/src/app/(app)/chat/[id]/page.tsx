"use client";

import { useState, useCallback } from "react";
import { ChatInput } from "@/components/chat/ChatInput";
import { ChatThread } from "@/components/chat/ChatThread";
import { MeshBackground } from "@/components/background/MeshBackground";
import { FeatureCycleButton } from "@/components/chat/FeatureCycleButton";
import { useAppStore } from "@/store/useAppStore";
import { mockMessages, type Message } from "@/lib/mockData";
import { useParams } from "next/navigation";

export default function ChatThreadPage() {
  const params = useParams();
  const chatId = params.id as string;
  const { chats } = useAppStore();

  const chat = chats.find((c) => c.id === chatId);
  const initialMessages = mockMessages.filter((m) => m.chatId === chatId);

  const [messages, setMessages] = useState<Message[]>(
    initialMessages.length > 0
      ? initialMessages
      : mockMessages.filter((m) => m.chatId === "c1") // fallback to first chat
  );
  const [isStreaming, setIsStreaming] = useState(false);

  const simulateResponse = useCallback((userMessage: string) => {
    setIsStreaming(true);
    const responseText = `Based on your request, I've run a comprehensive analysis:

**Lexical Analysis**
- Vocabulary richness (TTR): 0.68
- Average word length: 4.8 characters
- Formal vs informal ratio: 0.3

**Syntactic Patterns**
- Sentence complexity: Medium
- Active voice preference: 78%
- Question frequency: Low

**Summary**
The writing exhibits a balanced, accessible style. Shall I proceed with a style transfer or generate a comparison profile?`;

    const assistantMsg: Message = {
      id: `msg-${Date.now()}-resp`,
      chatId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, assistantMsg]);

    let charIndex = 0;
    const interval = setInterval(() => {
      charIndex += 2;
      if (charIndex >= responseText.length) {
        charIndex = responseText.length;
        clearInterval(interval);
        setIsStreaming(false);
      }
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id
            ? { ...m, content: responseText.slice(0, charIndex) }
            : m
        )
      );
    }, 15);
  }, [chatId]);

  const handleSend = (content: string) => {
    const userMsg: Message = {
      id: `msg-${Date.now()}`,
      chatId,
      role: "user",
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setTimeout(() => simulateResponse(content), 500);
  };

  return (
    <div className="relative flex flex-col h-screen">
      <MeshBackground
        colors={["#000000", "#0d0d1a", "#110a1f", "#0a0a0a"]}
        speed={0.3}
        className="absolute inset-0 w-full h-full"
      />

      {/* Top bar */}
      <div className="relative z-10 flex items-center justify-center py-3 min-h-[52px] border-b border-white/[0.04]">
        <div className="absolute left-14 lg:left-4 top-1/2 -translate-y-1/2">
          <FeatureCycleButton />
        </div>
        <span className="text-sm text-white/40">{chat?.title || "Chat"}</span>
      </div>

      {/* Messages */}
      <div className="relative z-10 flex-1 overflow-hidden">
        <ChatThread messages={messages} />
      </div>

      {/* Input */}
      <div className="relative z-10 p-4 pb-6">
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>
    </div>
  );
}
