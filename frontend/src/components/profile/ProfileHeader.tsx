"use client";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/store/useAppStore";

export function ProfileHeader() {
  const { user } = useAppStore();
  if (!user) return null;

  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-6">
      <div className="flex items-center gap-4 sm:gap-6">
        <div className="relative group">
          <Avatar className="h-16 w-16">
            <AvatarImage src={user.avatarUrl} alt={user.displayName} />
            <AvatarFallback className="text-lg">{user.displayName.slice(0, 2).toUpperCase()}</AvatarFallback>
          </Avatar>
          <div className="absolute inset-0 rounded-full bg-black/50 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity cursor-pointer">
            <span className="text-[10px] text-white font-medium">Edit</span>
          </div>
        </div>
        <div className="flex-1 min-w-0">
          <h2 className="text-xl font-bold text-white">{user.displayName}</h2>
          <p className="text-sm text-white/40">{user.username}</p>
          <p className="text-sm text-white/30">{user.email}</p>
        </div>
        <Button variant="outline" size="sm" className="hidden sm:flex">
          Edit
        </Button>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-6">
        <div className="rounded-xl border border-white/8 bg-white/[0.02] p-3">
          <p className="text-[10px] uppercase tracking-widest text-white/30 mb-1">Display Name</p>
          <p className="text-sm text-white font-medium">{user.displayName}</p>
        </div>
        <div className="rounded-xl border border-white/8 bg-white/[0.02] p-3">
          <p className="text-[10px] uppercase tracking-widest text-white/30 mb-1">Email</p>
          <p className="text-sm text-white font-medium">{user.email}</p>
        </div>
      </div>
    </div>
  );
}
