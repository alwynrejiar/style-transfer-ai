-- Run this script in your Supabase SQL Editor to enable self-serve account deletion.
-- Since the `auth.users` table is protected, normal users cannot delete their own rows directly.
-- This RPC function uses `SECURITY DEFINER` to bypass RLS safely, ensuring users can only delete their own account.

CREATE OR REPLACE FUNCTION delete_user()
RETURNS void AS $$
BEGIN
  -- Delete the user from the protected auth.users table.
  -- This will automatically cascade and delete the user's row in public.profiles.
  DELETE FROM auth.users WHERE id = auth.uid();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
