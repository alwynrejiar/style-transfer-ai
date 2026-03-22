"use client";

import { useState, useEffect, useCallback } from "react";
import { ChatInput } from "@/components/chat/ChatInput";
import { EmptyChat } from "@/components/chat/EmptyChat";
import { ChatThread } from "@/components/chat/ChatThread";
import { MeshBackground } from "@/components/background/MeshBackground";
import { useAppStore } from "@/store/useAppStore";
import type { Message } from "@/lib/mockData";

export default function ChatPage() {
  const { createChat, addMessage } = useAppStore();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const simulateResponse = useCallback((userMessage: string) => {
    setIsStreaming(true);
    const responseText = `I've analyzed your message. Here's what I found:

**Stylometric Profile**
- Writing style: Conversational
- Complexity score: 6.2/10
- Sentence variety: High

Your text shows a natural, conversational writing pattern with good rhythm and moderate complexity. The vocabulary richness indicates a well-developed personal voice.

Would you like me to perform a deeper analysis or transform this text into a different style?`;

    const assistantMsg: Message = {
      id: `msg-${Date.now()}-resp`,
      chatId: "new",
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
  }, []);

  const handleSend = (content: string) => {
    const userMsg: Message = {
      id: `msg-${Date.now()}`,
      chatId: "new",
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
        speed={0.4}
        className="absolute inset-0 w-full h-full"
      />

      {/* Top bar */}
      <div className="relative z-10 flex items-center justify-center py-3 border-b border-white/[0.04]">
        <span className="text-sm text-white/40">New Chat</span>
      </div>

      {/* Messages */}
      <div className="relative z-10 flex-1 overflow-hidden">
        {messages.length === 0 ? (
          <EmptyChat />
        ) : (
          <ChatThread messages={messages} />
        )}
      </div>

      {/* Input */}
      <div className="relative z-10 p-4 pb-6">
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>
    </div>
  );
}
