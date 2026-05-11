"""
Manual Supabase authentication smoke test.
Run: python tests/test_supabase_auth.py

This verifies only auth behavior. Application data is stored locally.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    print("=" * 60)
    print("  SUPABASE AUTH TEST")
    print("=" * 60)

    print("\n[1/5] Testing Supabase auth client...")
    try:
        from src.database.supabase_client import get_supabase_client

        get_supabase_client()
        print("  Connected to Supabase auth successfully.")
    except Exception as e:
        print(f"  Connection failed: {e}")
        print("  Make sure .env has SUPABASE_URL and SUPABASE_ANON_KEY set.")
        return

    print("\n[2/5] Testing sign up...")
    from src.database.auth import get_current_user, sign_in, sign_out, sign_up

    test_email = input("  Enter a test email: ").strip()
    test_password = input("  Enter a test password (min 6 chars): ").strip()
    test_name = input("  Enter a display name: ").strip() or "Test User"

    result = sign_up(test_email, test_password, test_name)
    if result["success"]:
        print(f"  Signed up. User ID: {result['data']['user_id']}")
        print("  If email confirmation is enabled, confirm the email before signing in.")
    else:
        print(f"  Sign up failed: {result['error']}")
        print("  If the user already exists, continue with sign in.")

    print("\n[3/5] Testing sign in...")
    result = sign_in(test_email, test_password)
    if not result["success"]:
        print(f"  Sign in failed: {result['error']}")
        return

    print(f"  Signed in. User ID: {result['data']['user_id']}")
    access_token = result["data"]["access_token"]

    print("\n[4/5] Testing current user lookup...")
    result = get_current_user(access_token)
    if result["success"]:
        print(f"  Email: {result['data']['email']}")
        print(f"  Name:  {result['data']['name']}")
    else:
        print(f"  Lookup failed: {result['error']}")

    print("\n[5/5] Testing sign out...")
    result = sign_out()
    if result["success"]:
        print("  Signed out successfully.")
    else:
        print(f"  Sign out failed: {result['error']}")

    print("\n" + "=" * 60)
    print("  AUTH TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
