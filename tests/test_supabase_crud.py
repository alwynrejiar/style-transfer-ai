"""
Test Supabase CRUD operations.
Run: python tests/test_supabase_crud.py

Prerequisites:
  - Run migration.sql in Supabase SQL Editor first
  - Have a confirmed user (run test_supabase_auth.py first)
  - .env file with SUPABASE_URL and SUPABASE_ANON_KEY

This will:
  1. Sign in
  2. Save a mock style analysis
  3. List analyses
  4. Fetch the full analysis
  5. Save mock generated content
  6. Save a mock style transfer
  7. Delete the test data
  8. Sign out
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# A small mock analysis that mirrors what the real analyzer produces
MOCK_ANALYSIS = {
    "metadata": {
        "model_used": "gemma3:1b",
        "processing_mode": "enhanced",
        "file_info": [
            {"filename": "test_sample.txt", "word_count": 500, "char_count": 2800}
        ],
    },
    "user_profile": {
        "name": "Test User",
        "native_language": "English",
        "english_fluency": "Native",
        "education_level": "Bachelor's",
        "writing_experience": "Intermediate",
    },
    "consolidated_analysis": {
        "lexical": {"vocabulary_tier": "intermediate", "signature_words": ["however", "therefore"]},
        "syntactic": {"avg_sentence_length": 18.5, "passive_voice_ratio": 0.12},
        "voice": {"formality_level": "semi-formal"},
    },
    "readability_metrics": {
        "flesch_reading_ease": 62.3,
        "flesch_kincaid_grade": 8.1,
        "gunning_fog": 10.2,
    },
    "confidence_report": {"overall": 0.82},
    "style_fingerprint_summary": "Semi-formal, mid-length sentences, moderate vocabulary.",
    "most_distinctive_traits": ["Frequent use of transitional phrases", "Balanced clause structure"],
    "key_traits": ["Semi-formal register", "Medium sentence length"],
    "rewrite_directive": "Maintain transitional phrases and semi-formal tone throughout.",
    "do_not_lose": ["Transitional phrases", "Measured pacing"],
    "avoid_in_rewrite": ["Overly casual slang", "Run-on sentences"],
    "cognitive_bridging": None,
}


def main():
    print("=" * 60)
    print("  SUPABASE CRUD TEST")
    print("=" * 60)

    from src.database.auth import sign_in, sign_out
    from src.database.db_analyses import save_analysis, list_analyses, get_analysis, delete_analysis
    from src.database.db_content import save_generated_content, list_generated_content, delete_generated_content
    from src.database.db_content import save_style_transfer, list_style_transfers, delete_style_transfer

    # ------------------------------------------------------------------
    # Step 1: Sign in
    # ------------------------------------------------------------------
    print("\n[1/8] Signing in...")
    email = input("  Email: ").strip()
    password = input("  Password: ").strip()

    auth = sign_in(email, password)
    if not auth["success"]:
        print(f"  ✗ Sign in failed: {auth['error']}")
        return

    token = auth["data"]["access_token"]
    uid = auth["data"]["user_id"]
    print(f"  ✓ Signed in as {auth['data']['email']}")

    created_ids = {"analysis": None, "content": None, "transfer": None}

    try:
        # ------------------------------------------------------------------
        # Step 2: Save analysis
        # ------------------------------------------------------------------
        print("\n[2/8] Saving mock style analysis...")
        result = save_analysis(token, uid, MOCK_ANALYSIS)
        if result["success"]:
            created_ids["analysis"] = result["data"]["id"]
            print(f"  ✓ Analysis saved! ID: {created_ids['analysis']}")
        else:
            print(f"  ✗ Failed: {result['error']}")
            return

        # ------------------------------------------------------------------
        # Step 3: List analyses
        # ------------------------------------------------------------------
        print("\n[3/8] Listing analyses...")
        result = list_analyses(token, uid)
        if result["success"]:
            print(f"  ✓ Found {len(result['data'])} analysis(es):")
            for a in result["data"]:
                print(f"    • {a['analysis_name']} ({a['processing_mode']}) — {a['created_at']}")
        else:
            print(f"  ✗ Failed: {result['error']}")

        # ------------------------------------------------------------------
        # Step 4: Get full analysis
        # ------------------------------------------------------------------
        print("\n[4/8] Fetching full analysis...")
        result = get_analysis(token, uid, created_ids["analysis"])
        if result["success"]:
            data = result["data"]
            print(f"  ✓ Full analysis retrieved:")
            print(f"    Name: {data['analysis_name']}")
            print(f"    Model: {data['model_used']}")
            print(f"    Fingerprint: {data['style_fingerprint_summary']}")
            print(f"    Traits: {data['most_distinctive_traits']}")
        else:
            print(f"  ✗ Failed: {result['error']}")

        # ------------------------------------------------------------------
        # Step 5: Save generated content
        # ------------------------------------------------------------------
        print("\n[5/8] Saving mock generated content...")
        content_data = {
            "style_analysis_id": created_ids["analysis"],
            "content_type": "article",
            "topic": "Test: The Future of AI Writing",
            "content": "This is a test article generated during CRUD testing.",
            "target_length": 500,
            "actual_length": 52,
            "tone": "professional",
            "model_used": "gemma3:1b",
            "quality_metrics": {"word_count": 10, "coherence_score": 0.85},
            "style_adherence_score": 0.78,
        }
        result = save_generated_content(token, uid, content_data)
        if result["success"]:
            created_ids["content"] = result["data"]["id"]
            print(f"  ✓ Content saved! ID: {created_ids['content']}")
        else:
            print(f"  ✗ Failed: {result['error']}")

        # ------------------------------------------------------------------
        # Step 6: Save style transfer
        # ------------------------------------------------------------------
        print("\n[6/8] Saving mock style transfer...")
        transfer_data = {
            "style_analysis_id": created_ids["analysis"],
            "transfer_type": "direct_transfer",
            "intensity": 0.8,
            "original_content": "The quick brown fox jumps over the lazy dog.",
            "transferred_content": "A swift, tawny fox leaps gracefully over the languorous canine.",
            "preserve_elements": ["meaning", "tone"],
            "model_used": "gemma3:1b",
            "style_match_score": 0.82,
        }
        result = save_style_transfer(token, uid, transfer_data)
        if result["success"]:
            created_ids["transfer"] = result["data"]["id"]
            print(f"  ✓ Transfer saved! ID: {created_ids['transfer']}")
        else:
            print(f"  ✗ Failed: {result['error']}")

        # ------------------------------------------------------------------
        # Step 7: List content & transfers
        # ------------------------------------------------------------------
        print("\n[7/8] Listing content & transfers...")
        result = list_generated_content(token, uid)
        if result["success"]:
            print(f"  ✓ Generated content: {len(result['data'])} item(s)")

        result = list_style_transfers(token, uid)
        if result["success"]:
            print(f"  ✓ Style transfers: {len(result['data'])} item(s)")

        # ------------------------------------------------------------------
        # Step 8: Cleanup test data
        # ------------------------------------------------------------------
        print("\n[8/8] Cleaning up test data...")

        if created_ids["content"]:
            result = delete_generated_content(token, uid, created_ids["content"])
            print(f"  {'✓' if result['success'] else '✗'} Delete content: {result.get('error', 'OK')}")

        if created_ids["transfer"]:
            result = delete_style_transfer(token, uid, created_ids["transfer"])
            print(f"  {'✓' if result['success'] else '✗'} Delete transfer: {result.get('error', 'OK')}")

        if created_ids["analysis"]:
            result = delete_analysis(token, uid, created_ids["analysis"])
            print(f"  {'✓' if result['success'] else '✗'} Delete analysis: {result.get('error', 'OK')}")

    finally:
        # Always sign out
        sign_out()
        print("\n  Signed out.")

    print("\n" + "=" * 60)
    print("  CRUD TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
