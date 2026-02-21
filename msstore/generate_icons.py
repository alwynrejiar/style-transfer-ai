"""
Generate all required icon sizes for the Microsoft Store MSIX package
from a single source image.

Usage:
    pip install Pillow
    python msstore/generate_icons.py [source_image]

If no source image is given, it generates placeholder icons with the app
logo text so you can test the packaging pipeline immediately.

Required MSIX icon sizes:
    app.ico           — Multi-resolution .ico for the .exe (16,32,48,64,128,256)
    Square44x44Logo   — 44x44   (app list)
    Square71x71Logo   — 71x71   (medium tile)
    Square150x150Logo — 150x150 (medium tile)
    Square310x310Logo — 310x310 (large tile)
    Wide310x150Logo   — 310x150 (wide tile)
    StoreLogo         — 50x50   (store listing)
    SplashScreen      — 620x300 (splash)
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
ICONS_DIR = SCRIPT_DIR / "icons"
MSIX_ASSETS_DIR = ICONS_DIR / "msix"

# Background color matching the app theme
BG_COLOR = (26, 26, 46)       # #1a1a2e
ACCENT_COLOR = (108, 92, 231)  # #6C5CE7
TEXT_COLOR = (255, 255, 255)

SIZES = {
    "Square44x44Logo.png": (44, 44),
    "Square71x71Logo.png": (71, 71),
    "Square150x150Logo.png": (150, 150),
    "Square310x310Logo.png": (310, 310),
    "Wide310x150Logo.png": (310, 150),
    "StoreLogo.png": (50, 50),
    "SplashScreen.png": (620, 300),
}

ICO_DIMS = [16, 32, 48, 64, 128, 256]


def _try_load_font(size: int):
    """Try to load a font, falling back to default."""
    for name in ["segoeui.ttf", "arial.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _make_placeholder(width: int, height: int) -> Image.Image:
    """Create a branded placeholder icon."""
    img = Image.new("RGBA", (width, height), BG_COLOR + (255,))
    draw = ImageDraw.Draw(img)

    # Draw accent circle/ellipse
    min_dim = min(width, height)
    circle_r = int(min_dim * 0.35)
    cx, cy = width // 2, height // 2
    draw.ellipse(
        [cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
        fill=ACCENT_COLOR + (255,),
    )

    # Draw "S" letter in the circle
    font_size = max(12, int(circle_r * 1.2))
    font = _try_load_font(font_size)
    text = "S"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2 - bbox[1]), text, fill=TEXT_COLOR, font=font)

    # If wide enough, add "tylomex" after the circle
    if width >= 200:
        label_font = _try_load_font(max(12, int(min_dim * 0.12)))
        label = "tylomex"
        draw.text((cx + circle_r + 8, cy - int(min_dim * 0.06)), label, fill=TEXT_COLOR, font=label_font)

    return img


def generate_from_source(source_path: str):
    """Resize a source image into all required sizes."""
    src = Image.open(source_path).convert("RGBA")
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    MSIX_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate MSIX PNGs
    for name, (w, h) in SIZES.items():
        resized = src.resize((w, h), Image.LANCZOS)
        resized.save(MSIX_ASSETS_DIR / name, "PNG")
        print(f"  Created {name} ({w}x{h})")

    # Generate .ico
    ico_images = [src.resize((s, s), Image.LANCZOS) for s in ICO_DIMS]
    ico_path = ICONS_DIR / "app.ico"
    ico_images[0].save(ico_path, format="ICO", sizes=[(s, s) for s in ICO_DIMS], append_images=ico_images[1:])
    print(f"  Created app.ico ({', '.join(f'{s}x{s}' for s in ICO_DIMS)})")


def generate_placeholders():
    """Generate branded placeholder icons."""
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    MSIX_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate MSIX PNGs
    for name, (w, h) in SIZES.items():
        img = _make_placeholder(w, h)
        img.save(MSIX_ASSETS_DIR / name, "PNG")
        print(f"  Created {name} ({w}x{h})")

    # Generate .ico from square placeholders
    ico_images = [_make_placeholder(s, s) for s in ICO_DIMS]
    ico_path = ICONS_DIR / "app.ico"
    ico_images[0].save(ico_path, format="ICO", sizes=[(s, s) for s in ICO_DIMS], append_images=ico_images[1:])
    print(f"  Created app.ico ({', '.join(f'{s}x{s}' for s in ICO_DIMS)})")

    print(f"\nAll icons saved to: {ICONS_DIR}")
    print(f"MSIX assets saved to: {MSIX_ASSETS_DIR}")
    print("\nReplace these with your real logo by running:")
    print("  python msstore/generate_icons.py path/to/your_logo.png")


def main():
    print("=== Stylomex Icon Generator ===\n")
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        print(f"Generating icons from: {sys.argv[1]}")
        generate_from_source(sys.argv[1])
    else:
        print("No source image provided — generating branded placeholders.")
        generate_placeholders()


if __name__ == "__main__":
    main()
