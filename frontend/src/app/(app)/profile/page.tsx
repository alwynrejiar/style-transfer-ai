"use client";

import { motion } from "framer-motion";
import { ProfileHeader } from "@/components/profile/ProfileHeader";
import { ActiveMode } from "@/components/profile/ActiveMode";
import { UsageSummary } from "@/components/profile/UsageSummary";
import { SavedModes } from "@/components/profile/SavedModes";
import { RecentActivity } from "@/components/profile/RecentActivity";

export default function ProfilePage() {
  return (
    <div className="min-h-screen py-8 sm:py-12 px-4 sm:px-6">
      <div className="max-w-[900px] mx-auto space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0 }}
        >
          <ProfileHeader />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.08 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          <ActiveMode />
          <UsageSummary />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.16 }}
        >
          <SavedModes />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.24 }}
        >
          <RecentActivity />
        </motion.div>
      </div>
    </div>
  );
}
