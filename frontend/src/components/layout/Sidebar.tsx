"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import {
  Plus, Search, Wand2, GitCompare, FileSearch, GraduationCap, Settings, LogOut, Phone, Menu,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Sheet, SheetContent, SheetTrigger, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/useAppStore";
import { useAuth } from "@/hooks/useAuth";
import { useState } from "react";

const primaryNav = [
  { icon: Plus, label: "New Chat", href: "/chat" },
  { icon: Search, label: "Search Chat", href: "/chat?search=true" },
  { icon: Wand2, label: "Style Transformation", href: "/chat?mode=transfer" },
  { icon: GitCompare, label: "Style Comparison", href: "/chat?mode=compare" },
  { icon: FileSearch, label: "Style Analysis", href: "/chat?mode=analyze" },
  { icon: GraduationCap, label: "Student Analogy", href: "/chat?mode=student-analogy" },
];

const bottomNav = [
  { icon: Settings, label: "Settings", href: "/settings" },
  { icon: Phone, label: "Contact", href: "/contact" },
];

function SidebarContent({ pathname }: { pathname: string }) {
  const { user, chats } = useAppStore();
  const { signOut } = useAuth();

  const handleLogout = async () => {
    await signOut();
  };

  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="p-4 flex items-center gap-2.5">
        <Link href="/chat" className="flex items-center gap-2.5">
          <Image src="/logo.png" alt="Stylomex" width={28} height={28} className="rounded-md" />
          <span className="font-semibold text-white text-base tracking-tight">Stylomex.AI</span>
        </Link>
      </div>

      <div className="mx-3 border-t border-white/8" />

      {/* Primary Nav */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        {primaryNav.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href.split("?")[0]);
          return (
            <Link
              key={item.label}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors duration-150",
                isActive
                  ? "bg-white/10 text-white border-l-2 border-[var(--accent)]"
                  : "text-white/60 hover:bg-white/[0.06] hover:text-white"
              )}
            >
              <item.icon className="h-4 w-4 shrink-0" />
              <span className="truncate">{item.label}</span>
            </Link>
          );
        })}

        {/* Previous Chats */}
        <div className="pt-4">
          <div className="mx-3 border-t border-white/8 mb-3" />
          <p className="px-3 text-[10px] font-medium uppercase tracking-[0.15em] text-white/30 mb-2">
            Previous Chats
          </p>
          {chats.map((chat) => (
            <Link
              key={chat.id}
              href={`/chat/${chat.id}`}
              className={cn(
                "flex items-center gap-3 px-3 py-1.5 rounded-lg text-sm transition-colors duration-150",
                pathname === `/chat/${chat.id}`
                  ? "bg-white/10 text-white"
                  : "text-white/40 hover:bg-white/[0.06] hover:text-white/70"
              )}
            >
              <span className="truncate">{chat.title}</span>
            </Link>
          ))}
        </div>
      </nav>

      {/* Bottom Section */}
      <div className="mt-auto">
        <div className="mx-3 border-t border-white/8" />
        <div className="p-3 space-y-1">
          {bottomNav.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.label}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors duration-150",
                  isActive
                    ? "bg-white/10 text-white"
                    : "text-white/60 hover:bg-white/[0.06] hover:text-white"
                )}
              >
                <item.icon className="h-4 w-4 shrink-0" />
                <span className="truncate">{item.label}</span>
              </Link>
            );
          })}
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-white/60 hover:bg-white/[0.06] hover:text-white transition-colors duration-150 w-full cursor-pointer"
          >
            <LogOut className="h-4 w-4 shrink-0" />
            <span className="truncate">Logout</span>
          </button>
        </div>

        <div className="mx-3 border-t border-white/8" />

        {/* User Card */}
        {user && (
          <div className="p-3 flex items-center gap-3">
            <Avatar className="h-9 w-9">
              <AvatarImage src={user.avatarUrl} alt={user.displayName} />
              <AvatarFallback>{user.displayName.slice(0, 2).toUpperCase()}</AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">{user.displayName}</p>
              <p className="text-xs text-white/40 truncate">{user.email}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex fixed left-0 top-0 bottom-0 w-[240px] bg-[#111111] border-r border-white/8 z-40 flex-col">
        <SidebarContent pathname={pathname} />
      </aside>

      {/* Mobile hamburger */}
      <div className="lg:hidden fixed top-3 left-3 z-50">
        <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="text-white/60">
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="p-0 w-[280px]">
            <SheetTitle className="sr-only">Navigation</SheetTitle>
            <SheetDescription className="sr-only">Application navigation sidebar</SheetDescription>
            <SidebarContent pathname={pathname} />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
