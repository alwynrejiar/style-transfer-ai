import { create } from "zustand";
import { mockChats, mockModes, mockMessages, type User, type Chat, type Message, type Mode } from "@/lib/mockData";
interface ModelConfig {
  useLocal: boolean;
  modelName: string;
  geminiKey: string;
}

interface AnalogyConfig {
  enabled: boolean;
  defaultDomain: string;
}

interface AppStore {
  // Auth
  user: User | null;
  setUser: (user: User | null) => void;

  // Chat
  chats: Chat[];
  activeChat: string | null;
  setActiveChat: (chatId: string | null) => void;
  messages: Message[];
  addMessage: (chatId: string, message: Omit<Message, "id">) => void;
  createChat: (title?: string) => string;

  // Profile
  modes: Mode[];
  activeMode: string;
  setActiveMode: (mode: string) => void;

  // Settings
  modelConfig: ModelConfig;
  updateModelConfig: (config: Partial<ModelConfig>) => void;
  analogyConfig: AnalogyConfig;
  updateAnalogyConfig: (config: Partial<AnalogyConfig>) => void;
  theme: "dark" | "light" | "system";
  setTheme: (theme: "dark" | "light" | "system") => void;
}

export const useAppStore = create<AppStore>((set) => ({
  // Auth
  user: null,
  setUser: (user) => set({ user }),

  // Chat
  chats: mockChats,
  activeChat: null,
  setActiveChat: (chatId) => set({ activeChat: chatId }),
  messages: mockMessages,
  addMessage: (chatId, message) => {
    const id = `msg-${Date.now()}`;
    set((state) => ({
      messages: [...state.messages, { ...message, id }],
    }));
  },
  createChat: (title) => {
    const id = `chat-${Date.now()}`;
    const newChat: Chat = {
      id,
      title: title || "New Chat",
      updatedAt: "Just now",
    };
    set((state) => ({
      chats: [newChat, ...state.chats],
      activeChat: id,
    }));
    return id;
  },

  // Profile
  modes: mockModes,
  activeMode: "Academic",
  setActiveMode: (mode) =>
    set((state) => ({
      activeMode: mode,
      modes: state.modes.map((m) => ({
        ...m,
        active: m.name === mode,
      })),
    })),

  // Settings
  modelConfig: {
    useLocal: true,
    modelName: "gemma3:1b",
    geminiKey: "",
  },
  updateModelConfig: (config) =>
    set((state) => ({
      modelConfig: { ...state.modelConfig, ...config },
    })),
  analogyConfig: {
    enabled: true,
    defaultDomain: "General Simplification",
  },
  updateAnalogyConfig: (config) =>
    set((state) => ({
      analogyConfig: { ...state.analogyConfig, ...config },
    })),
  theme: "dark",
  setTheme: (theme) => set({ theme }),
}));
