"use client";

import { motion } from "framer-motion";
import { Mail, MapPin, Phone, Github, Twitter, Linkedin, Instagram } from "lucide-react";

const APP_EMAIL = "hello@stylomex.ai";
const APP_PHONE = "+91 98765 43210";

const contactInfo = [
  { icon: Mail, label: "Email", value: APP_EMAIL, href: `mailto:${APP_EMAIL}` },
  { icon: MapPin, label: "Office", value: "Kerala, India" },
  { icon: Phone, label: "Phone", value: APP_PHONE, href: `tel:${APP_PHONE}` },
];

const socialLinks = [
  { icon: Github, label: "GitHub", href: "https://github.com/alwynrejiar/style-transfer-ai" },
  { icon: Twitter, label: "Twitter", href: "https://twitter.com/stylomex" },
  { icon: Linkedin, label: "LinkedIn", href: "https://linkedin.com/company/stylomex" },
  { icon: Instagram, label: "Instagram", href: "https://instagram.com/stylomex" },
];

export default function ContactPage() {
  return (
    <div className="min-h-screen py-8 sm:py-12 px-4 sm:px-6">
      <div className="max-w-[900px] mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-white mb-2">Get in Touch</h1>
          <p className="text-white/40">We&apos;d love to hear from you. Reach out through any of the channels below.</p>
        </motion.div>

        {/* Contact Info Cards */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.08 }}
          className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8"
        >
          {contactInfo.map((item) => (
            <div
              key={item.label}
              className="rounded-2xl border border-white/8 bg-white/[0.04] p-6 text-center"
            >
              <div className="h-10 w-10 rounded-xl bg-[var(--accent)]/10 flex items-center justify-center mx-auto mb-3">
                <item.icon className="h-5 w-5 text-[var(--accent)]" />
              </div>
              <p className="text-xs uppercase tracking-widest text-white/30 mb-1">{item.label}</p>
              {item.href ? (
                <a
                  href={item.href}
                  className="text-sm text-white font-medium hover:text-[var(--accent)] transition-colors"
                >
                  {item.value}
                </a>
              ) : (
                <p className="text-sm text-white font-medium">{item.value}</p>
              )}
            </div>
          ))}
        </motion.div>

        {/* Social Links */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.16 }}
          className="rounded-2xl border border-white/8 bg-white/[0.04] p-6"
        >
          <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-white/30 mb-4">
            Follow Us
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {socialLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-3 rounded-xl border border-white/8 bg-white/[0.02] p-4 hover:bg-white/[0.06] hover:border-white/16 transition-all duration-200"
              >
                <link.icon className="h-5 w-5 text-white/40" />
                <span className="text-sm text-white/60 font-medium">{link.label}</span>
              </a>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
