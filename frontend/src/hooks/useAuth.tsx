"use client";

import { createContext, useContext, useEffect, useState, useCallback } from "react";
import { createClient } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import type { User, Session } from "@supabase/supabase-js";
import { useAppStore } from "@/store/useAppStore";

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<{ error: string | null }>;
  signUp: (email: string, password: string, name: string) => Promise<{ error: string | null }>;
  signOut: () => Promise<void>;
  deleteAccount: () => Promise<{ error: string | null }>;
  signInWithOAuth: (provider: "github" | "google") => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const supabase = createClient();

  const clearPersistedAuthState = useCallback(() => {
    // Supabase stores session tokens in browser storage under sb-*-auth-token keys.
    if (typeof window === "undefined") return;

    const clearStorage = (storage: Storage) => {
      const keysToRemove: string[] = [];
      for (let i = 0; i < storage.length; i += 1) {
        const key = storage.key(i);
        if (!key) continue;
        if ((key.startsWith("sb-") && key.includes("-auth-token")) || key.includes("supabase.auth.token")) {
          keysToRemove.push(key);
        }
      }
      keysToRemove.forEach((key) => storage.removeItem(key));
    };

    try {
      clearStorage(window.localStorage);
      clearStorage(window.sessionStorage);
    } catch (err) {
      console.warn("Failed to clear persisted auth storage:", err);
    }

    setSession(null);
    setUser(null);
    useAppStore.getState().setUser(null);
  }, []);

  useEffect(() => {
    // Get initial session
    const syncUser = async (session: Session | null) => {
      if (!session?.user) {
        useAppStore.getState().setUser(null);
        return;
      }
      
      try {
        const { data: profile } = await supabase
          .from("profiles")
          .select("*")
          .eq("id", session.user.id)
          .single();

        const name = profile?.name || session.user.email?.split('@')[0] || "User";
        
        useAppStore.getState().setUser({
          id: session.user.id,
          displayName: name,
          username: `@${name.toLowerCase().replace(/\\s+/g, "")}`,
          email: session.user.email || "",
          avatarUrl: `https://api.dicebear.com/7.x/notionists/svg?seed=${session.user.id}`,
        });
      } catch (err) {
        console.error("Failed to fetch user profile", err);
      }
    };

    const getSession = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        setSession(session);
        setUser(session?.user ?? null);
        await syncUser(session);
      } catch {
        console.warn("Supabase auth not available, using mock mode");
      } finally {
        setLoading(false);
      }
    };

    getSession();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        syncUser(session);
      }
    );

    return () => {
      subscription.unsubscribe();
    };
  }, [supabase]);

  const signIn = useCallback(
    async (email: string, password: string) => {
      try {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) return { error: error.message };
        router.push("/chat");
        return { error: null };
      } catch (e: any) {
        return { error: e.message || "Sign in failed" };
      }
    },
    [supabase, router]
  );

  const signUp = useCallback(
    async (email: string, password: string, name: string) => {
      try {
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: { name },
          },
        });
        if (error) return { error: error.message };
        router.push("/chat");
        return { error: null };
      } catch (e: any) {
        return { error: e.message || "Sign up failed" };
      }
    },
    [supabase, router]
  );

  const signOut = useCallback(async () => {
    try {
      await supabase.auth.signOut();
    } catch {
      // ignore errors on sign-out
    }
    clearPersistedAuthState();
    router.push("/login");
  }, [supabase, router, clearPersistedAuthState]);

  const deleteAccount = useCallback(async () => {
    try {
      const { error } = await supabase.rpc("delete_user");
      if (error) return { error: error.message };
      
      // We wrap signOut in a try/catch because the backend user is already gone,
      // so the session token is invalid and the API will reject the logout request.
      try {
        await supabase.auth.signOut();
      } catch (err) {
        console.warn("Expected error on sign out after deletion:", err);
      }

      clearPersistedAuthState();
      router.push("/login");
      return { error: null };
    } catch (e: any) {
      return { error: e.message || "Account deletion failed" };
    }
  }, [supabase, router, clearPersistedAuthState]);

  const signInWithOAuth = useCallback(
    async (provider: "github" | "google") => {
      try {
        await supabase.auth.signInWithOAuth({
          provider,
          options: {
            redirectTo: `${window.location.origin}/chat`,
            queryParams: provider === "google" ? { prompt: "select_account" } : undefined,
          },
        });
      } catch (e: any) {
        console.error("OAuth sign-in failed:", e.message);
      }
    },
    [supabase]
  );

  return (
    <AuthContext.Provider
      value={{ user, session, loading, signIn, signUp, signOut, deleteAccount, signInWithOAuth }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
