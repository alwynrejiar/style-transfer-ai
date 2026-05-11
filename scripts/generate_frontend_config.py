"""Generate browser-safe runtime config for the static app shell.

This script reads .env values and writes app/js/config.js so the frontend can
access Supabase auth public settings without hardcoding them in index.html.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import dotenv_values


def build_config(env_file: Path, output_file: Path) -> None:
    values = dotenv_values(env_file)
    supabase_url = (values.get("SUPABASE_URL") or "").strip()
    supabase_anon_key = (values.get("SUPABASE_ANON_KEY") or "").strip()

    if not supabase_url or not supabase_anon_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set before generating frontend config"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(
            [
                "window.SUPABASE_URL = " + repr(supabase_url) + ";",
                "window.SUPABASE_ANON_KEY = " + repr(supabase_anon_key) + ";",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate app/js/config.js from .env")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--out", default="app/js/config.js", help="Output JS file")
    args = parser.parse_args()

    env_file = Path(args.env)
    out_file = Path(args.out)

    if not env_file.exists():
        raise FileNotFoundError(f"Missing env file: {env_file}")

    build_config(env_file, out_file)
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
