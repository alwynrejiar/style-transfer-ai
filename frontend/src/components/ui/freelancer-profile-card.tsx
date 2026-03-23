"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { Star } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

/**
 * Props for the FreelancerProfileCard component.
 */
interface FreelancerProfileCardProps {
  /** The user's full name. */
  name: string;
  /** The user's job title or role. */
  title: string;
  /** The user's rating (e.g., 4.0). */
  rating: number;
  /** A string describing the project duration (e.g., "8 Days"). */
  duration: string;
  /** A string for the user's rate (e.g., "$40/hr"). */
  rate: string;
  /** A React node (e.g., array of icons) for the tools section. */
  tools: React.ReactNode;
  /** Optional click handler for the "Get in touch" button. */
  onGetInTouch?: () => void;
  /** Optional additional class names. */
  className?: string;
}

// Animation variants for Framer Motion
const cardVariants = {
  initial: { opacity: 0, y: 20 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: "easeOut" as const },
  },
  hover: {
    scale: 1.02,
    transition: { duration: 0.3 },
  },
};

const contentVariants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
};

const itemVariants = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3 } },
};

/**
 * A reusable, animated freelancer profile card component (no banner/avatar image).
 */
export const FreelancerProfileCard = React.forwardRef<
  HTMLDivElement,
  FreelancerProfileCardProps
>(
  (
    {
      className,
      name,
      title,
      rating,
      duration,
      rate,
      tools,
      onGetInTouch,
    },
    ref
  ) => {
    return (
      <motion.div
        ref={ref}
        className={cn(
          "relative w-full overflow-hidden rounded-2xl bg-card shadow-lg",
          className
        )}
        variants={cardVariants}
        initial="initial"
        animate="animate"
        whileHover="hover"
      >
        {/* Content Area */}
        <motion.div className="px-6 pb-6 pt-6" variants={contentVariants}>
          {/* Name, Title, and Tools */}
          <motion.div
            className="mb-4 flex items-start justify-between"
            variants={itemVariants}
          >
            <div>
              <h2 className="text-xl font-semibold text-card-foreground">
                {name}
              </h2>
              <p className="text-sm text-muted-foreground">{title}</p>
            </div>
            <div className="flex flex-col items-end gap-1.5">
              <div className="flex gap-1.5">{tools}</div>
              <span className="text-xs text-muted-foreground">Tools</span>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            className="my-4 flex items-center justify-around rounded-lg border border-border bg-background/30 p-4"
            variants={itemVariants}
          >
            <StatItem icon={Star} value={rating.toFixed(1)} label="Rating" />
            <Divider />
            <StatItem value={duration} label="Duration" />
            <Divider />
            <StatItem value={rate} label="Rate" />
          </motion.div>

          {/* Action Button */}
          <motion.div variants={itemVariants}>
            <Button className="w-full" size="lg" onClick={onGetInTouch}>
              Get in touch
            </Button>
          </motion.div>
        </motion.div>
      </motion.div>
    );
  }
);
FreelancerProfileCard.displayName = "FreelancerProfileCard";

// Internal StatItem component
const StatItem = ({
  icon: Icon,
  value,
  label,
}: {
  icon?: React.ElementType;
  value: string | number;
  label: string;
}) => (
  <div className="flex flex-1 flex-col items-center justify-center px-2 text-center">
    <div className="flex items-center gap-1">
      {Icon && <Icon className="h-4 w-4 text-muted-foreground" />}
      <span className="text-base font-semibold text-card-foreground">
        {value}
      </span>
    </div>
    <span className="text-xs capitalize text-muted-foreground">{label}</span>
  </div>
);

// Internal Divider component
const Divider = () => <div className="h-10 w-px bg-border" />;
