-- =============================================================================
-- Style Transfer AI — Supabase Database Migration
-- =============================================================================
-- Run this ENTIRE script in: Supabase Dashboard → SQL Editor → New Query
-- It creates all tables, enables Row Level Security, and sets up policies.
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. PROFILES — extends auth.users with user demographic/writing data
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.profiles (
    id              UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    name            TEXT,
    native_language TEXT,
    english_fluency TEXT CHECK (english_fluency IN ('Beginner', 'Intermediate', 'Advanced', 'Native')),
    other_languages TEXT,
    nationality     TEXT,
    cultural_background TEXT,
    education_level TEXT CHECK (education_level IN ('High School', 'Bachelor''s', 'Master''s', 'PhD', 'Other')),
    field_of_study  TEXT,
    writing_experience TEXT CHECK (writing_experience IN ('Beginner', 'Intermediate', 'Advanced', 'Professional')),
    writing_frequency TEXT CHECK (writing_frequency IN ('Daily', 'Weekly', 'Monthly', 'Rarely')),
    preferred_model TEXT DEFAULT 'gemma3:1b',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Auto-update updated_at on every UPDATE
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_profiles_updated_at
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();

-- Auto-create a profiles row when a new user signs up via auth
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, name)
    VALUES (
        NEW.id,
        COALESCE(NEW.raw_user_meta_data ->> 'name', NEW.email)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. STYLE ANALYSES — main analysis results (replaces JSON files)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.style_analyses (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                   UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    analysis_name             TEXT NOT NULL,
    processing_mode           TEXT CHECK (processing_mode IN ('fast', 'statistical', 'enhanced')),
    model_used                TEXT,
    source_files              JSONB DEFAULT '[]'::jsonb,
    consolidated_analysis     JSONB DEFAULT '{}'::jsonb,
    readability_metrics       JSONB DEFAULT '{}'::jsonb,
    confidence_report         JSONB DEFAULT '{}'::jsonb,
    style_fingerprint_summary TEXT,
    most_distinctive_traits   JSONB DEFAULT '[]'::jsonb,
    key_traits                JSONB DEFAULT '[]'::jsonb,
    rewrite_directive         TEXT,
    do_not_lose               JSONB DEFAULT '[]'::jsonb,
    avoid_in_rewrite          JSONB DEFAULT '[]'::jsonb,
    cognitive_bridging        JSONB,
    created_at                TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_style_analyses_user_id ON public.style_analyses(user_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. GENERATED CONTENT — articles, emails, stories, etc.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.generated_content (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id              UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    style_analysis_id    UUID REFERENCES public.style_analyses(id) ON DELETE SET NULL,
    content_type         TEXT,
    topic                TEXT,
    content              TEXT,
    target_length        INTEGER,
    actual_length        INTEGER,
    tone                 TEXT,
    model_used           TEXT,
    quality_metrics      JSONB DEFAULT '{}'::jsonb,
    style_adherence_score DOUBLE PRECISION,
    created_at           TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_generated_content_user_id ON public.generated_content(user_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. STYLE TRANSFERS — transferred content
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.style_transfers (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    style_analysis_id   UUID REFERENCES public.style_analyses(id) ON DELETE SET NULL,
    transfer_type       TEXT CHECK (transfer_type IN (
        'direct_transfer', 'style_blend', 'gradual_transform',
        'tone_shift', 'formality_adjust', 'audience_adapt'
    )),
    intensity           DOUBLE PRECISION CHECK (intensity >= 0.0 AND intensity <= 1.0),
    original_content    TEXT,
    transferred_content TEXT,
    preserve_elements   JSONB DEFAULT '[]'::jsonb,
    model_used          TEXT,
    style_match_score   DOUBLE PRECISION,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_style_transfers_user_id ON public.style_transfers(user_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. STYLE COMPARISONS — comparison results between two profiles
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.style_comparisons (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    profile_a_id      UUID REFERENCES public.style_analyses(id) ON DELETE SET NULL,
    profile_b_id      UUID REFERENCES public.style_analyses(id) ON DELETE SET NULL,
    comparison_result JSONB DEFAULT '{}'::jsonb,
    similarity_score  DOUBLE PRECISION,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_style_comparisons_user_id ON public.style_comparisons(user_id);

-- =============================================================================
-- ROW LEVEL SECURITY — each user can only access their own data
-- =============================================================================

-- Enable RLS on all tables
ALTER TABLE public.profiles          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.style_analyses    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.generated_content ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.style_transfers   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.style_comparisons ENABLE ROW LEVEL SECURITY;

-- PROFILES: user can read/update their own row
CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

-- STYLE ANALYSES: full CRUD on own rows
CREATE POLICY "Users can view own analyses"
    ON public.style_analyses FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own analyses"
    ON public.style_analyses FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own analyses"
    ON public.style_analyses FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own analyses"
    ON public.style_analyses FOR DELETE
    USING (auth.uid() = user_id);

-- GENERATED CONTENT: full CRUD on own rows
CREATE POLICY "Users can view own generated content"
    ON public.generated_content FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own generated content"
    ON public.generated_content FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own generated content"
    ON public.generated_content FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own generated content"
    ON public.generated_content FOR DELETE
    USING (auth.uid() = user_id);

-- STYLE TRANSFERS: full CRUD on own rows
CREATE POLICY "Users can view own transfers"
    ON public.style_transfers FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own transfers"
    ON public.style_transfers FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own transfers"
    ON public.style_transfers FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own transfers"
    ON public.style_transfers FOR DELETE
    USING (auth.uid() = user_id);

-- STYLE COMPARISONS: full CRUD on own rows
CREATE POLICY "Users can view own comparisons"
    ON public.style_comparisons FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own comparisons"
    ON public.style_comparisons FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own comparisons"
    ON public.style_comparisons FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own comparisons"
    ON public.style_comparisons FOR DELETE
    USING (auth.uid() = user_id);

-- =============================================================================
-- DONE! You should see "Success. No rows returned" — that means it worked.
-- =============================================================================
