"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname, useSearchParams } from "next/navigation";
import {
  Plus, Search, Wand2, GitCompare, FileSearch, GraduationCap, Settings, LogOut, Phone, Menu,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Sheet, SheetContent, SheetTrigger, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/useAppStore";
import { useAuth } from "@/hooks/useAuth";
import { useState } from "react";
import { STYLE_FEATURES, useStyleFeatureCycle } from "@/hooks/useStyleFeatureCycle";

const coreNav = [
  { icon: Plus, label: "New Chat", href: "/chat" },
  { icon: Search, label: "Search Chat", href: "/chat?search=true" },
];

const styleFeatureIcons: Record<string, LucideIcon> = {
  transfer: Wand2,
  compare: GitCompare,
  analyze: FileSearch,
  "student-analogy": GraduationCap,
};

const bottomNav = [
  { icon: Settings, label: "Settings", href: "/settings" },
  { icon: Phone, label: "Contact", href: "/contact" },
];

function isNavItemActive(pathname: string, searchParams: ReturnType<typeof useSearchParams>, href: string) {
  const [hrefPath, queryString] = href.split("?");
  if (pathname !== hrefPath) return false;

  if (!queryString) {
    // "New Chat" should not appear active while another chat mode is selected.
    if (href === "/chat") {
      return !searchParams.get("mode") && !searchParams.get("search");
    }
    return true;
  }

  const expectedParams = new URLSearchParams(queryString);
  for (const [key, value] of expectedParams.entries()) {
    if (searchParams.get(key) !== value) return false;
  }
  return true;
}

function SidebarContent() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const { user, chats } = useAppStore();
  const { signOut } = useAuth();
  const { sidebarStyleFeatures, setHiddenFeatureIndex } = useStyleFeatureCycle();

  const primaryNav = [
    ...coreNav,
    ...sidebarStyleFeatures.map((feature) => ({
      icon: styleFeatureIcons[feature.key],
      label: feature.label,
      href: feature.href,
    })),
  ];

  const handleLogout = async () => {
    await signOut();
  };

  const handlePrimaryNavClick = (href: string) => {
    const clickedStyleFeatureIndex = STYLE_FEATURES.findIndex((feature) => feature.href === href);
    if (clickedStyleFeatureIndex >= 0) {
      // Swap positions: clicked sidebar style becomes the hidden top-left feature.
      setHiddenFeatureIndex(clickedStyleFeatureIndex);
    }
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
          const isActive = isNavItemActive(pathname, searchParams, item.href);
          return (
            <Link
              key={item.label}
              href={item.href}
              onClick={() => handlePrimaryNavClick(item.href)}
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
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex fixed left-0 top-0 bottom-0 w-[240px] bg-[#111111] border-r border-white/8 z-40 flex-col">
        <SidebarContent />
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
            <SidebarContent />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
