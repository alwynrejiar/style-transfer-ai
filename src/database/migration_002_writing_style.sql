-- =============================================================================
-- Style Transfer AI — Add Writing Style Identity to Profiles
-- =============================================================================
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- This adds content analysis / writing style columns to the existing profiles table.
-- Safe to run multiple times (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
-- =============================================================================

-- Add writing style columns to profiles
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS vocabulary_level       TEXT;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS avg_sentence_length    DOUBLE PRECISION;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS formality_level        TEXT;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS tone_profile           TEXT;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS readability_score      DOUBLE PRECISION;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS grade_level            DOUBLE PRECISION;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS style_fingerprint      TEXT;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS dominant_traits        JSONB DEFAULT '[]'::jsonb;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS writing_strengths      JSONB DEFAULT '[]'::jsonb;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS writing_weaknesses     JSONB DEFAULT '[]'::jsonb;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS preferred_content_types JSONB DEFAULT '[]'::jsonb;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS total_analyses_count   INTEGER DEFAULT 0;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS last_analysis_at       TIMESTAMPTZ;

-- =============================================================================
-- DONE! You should see "Success. No rows returned" — that means it worked.
-- =============================================================================
