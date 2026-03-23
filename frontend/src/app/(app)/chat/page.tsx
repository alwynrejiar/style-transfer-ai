"use client";

import { useState, useEffect, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { ChatInput } from "@/components/chat/ChatInput";
import { EmptyChat } from "@/components/chat/EmptyChat";
import { ChatThread } from "@/components/chat/ChatThread";
import { MeshBackground } from "@/components/background/MeshBackground";
import { FeatureCycleButton } from "@/components/chat/FeatureCycleButton";
import { useAppStore } from "@/store/useAppStore";
import { createClient } from "@/lib/supabase";
import { analyzeStyle } from "@/lib/analyzeStyle";
import { saveAnalysis } from "@/lib/saveAnalysis";
import type { Message } from "@/lib/mockData";

// Inner component that uses useSearchParams (must be inside Suspense)
function ChatPageInner() {
  const searchParams = useSearchParams();
  const mode = searchParams.get("mode");
  const isAnalyzeMode = mode === "analyze";

  const { user, modelConfig } = useAppStore();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [noKeyBanner, setNoKeyBanner] = useState(false);

  // Reset messages when mode changes
  useEffect(() => {
    setMessages([]);
    setNoKeyBanner(false);
  }, [mode]);

  const simulateResponse = useCallback((userMessage: string) => {
    setIsStreaming(true);
    const responseText = `I've analyzed your message. Here's what I found:\n\n**Stylometric Profile**\n- Writing style: Conversational\n- Complexity score: 6.2/10\n- Sentence variety: High\n\nYour text shows a natural, conversational writing pattern with good rhythm and moderate complexity. The vocabulary richness indicates a well-developed personal voice.\n\nWould you like me to perform a deeper analysis or transform this text into a different style?`;

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

  const runStyleAnalysis = useCallback(
    async (text: string) => {
      const apiKey = modelConfig.geminiKey;
      if (!apiKey) {
        setNoKeyBanner(true);
        setIsStreaming(false);
        return;
      }

      setNoKeyBanner(false);

      // Loading message
      const loadingId = `msg-${Date.now()}-loading`;
      const loadingMsg: Message = {
        id: loadingId,
        chatId: "new",
        role: "assistant",
        content: "Analyzing your writing style…",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, loadingMsg]);

      try {
        const result = await analyzeStyle(text, apiKey);

        let saved = false;
        if (user?.id) {
          const supabase = createClient();
          const saveResult = await saveAnalysis(
            supabase,
            user.id,
            user.displayName,
            result
          );
          saved = saveResult.success;
        }

        // Replace loading message with analysis card
        const analysisMsg: Message = {
          id: `msg-${Date.now()}-analysis`,
          chatId: "new",
          role: "assistant",
          content: saved ? "saved" : "unsaved",
          timestamp: new Date(),
          analysisResult: result,
        };

        setMessages((prev) =>
          prev.filter((m) => m.id !== loadingId).concat(analysisMsg)
        );
      } catch (err: unknown) {
        const errorMsg =
          err instanceof Error && err.message === "NO_API_KEY"
            ? "No Gemini API key found. Add your key in **Settings → Model Config**."
            : `Analysis failed: ${err instanceof Error ? err.message : "Unknown error"}`;

        setMessages((prev) =>
          prev
            .filter((m) => m.id !== loadingId)
            .concat({
              id: `msg-${Date.now()}-err`,
              chatId: "new",
              role: "assistant",
              content: errorMsg,
              timestamp: new Date(),
            })
        );
      } finally {
        setIsStreaming(false);
      }
    },
    [modelConfig.geminiKey, user]
  );

  const handleSend = (content: string) => {
    const userMsg: Message = {
      id: `msg-${Date.now()}`,
      chatId: "new",
      role: "user",
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);

    if (isAnalyzeMode) {
      runStyleAnalysis(content);
    } else {
      setTimeout(() => simulateResponse(content), 500);
    }
  };

  const modeLabel = (() => {
    switch (mode) {
      case "analyze": return "Style Analysis";
      case "transfer": return "Style Transformation";
      case "compare": return "Style Comparison";
      case "student-analogy": return "Style Analogy";
      default: return "New Chat";
    }
  })();

  const placeholder = isAnalyzeMode
    ? "Paste your text here to analyze writing style…"
    : "Type your message here…";

  return (
    <div className="relative flex flex-col h-screen">
      <MeshBackground
        colors={["#000000", "#0d0d1a", "#110a1f", "#0a0a0a"]}
        speed={0.4}
        className="absolute inset-0 w-full h-full"
      />

      {/* Top bar */}
      <div className="relative z-10 flex items-center justify-center py-3 min-h-[52px] border-b border-white/[0.04]">
        <div className="absolute left-14 lg:left-4 top-1/2 -translate-y-1/2">
          <FeatureCycleButton />
        </div>
        <span className="text-sm text-white/40">{modeLabel}</span>
      </div>

      {/* No-key banner */}
      {noKeyBanner && (
        <div className="relative z-10 mx-4 mt-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
          No Gemini API key found. Add your key in{" "}
          <a href="/settings" className="underline hover:text-amber-100">
            Settings → Model Config
          </a>
          .
        </div>
      )}

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
        <ChatInput
          onSend={handleSend}
          disabled={isStreaming}
          placeholder={placeholder}
        />
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={null}>
      <ChatPageInner />
    </Suspense>
  );
}
