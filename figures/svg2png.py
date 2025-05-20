#!/usr/bin/env python3
"""
svg2png.py

A small utility to convert SVG files to PNG format using CairoSVG.

Usage:
    python svg2png.py input.svg [output.png]

If you omit the output filename, it will default to the same basename with .png.
"""

import sys
import os
from cairosvg import svg2png

def convert(svg_path: str, png_path: str):
    try:
        svg2png(url=svg_path, write_to=png_path)
        print(f"✔ Converted '{svg_path}' → '{png_path}'")
    except Exception as e:
        print(f"✖ Failed to convert '{svg_path}': {e}", file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        print("Usage: svg2png.py input.svg [output.png]", file=sys.stderr)
        sys.exit(1)

    svg_file = sys.argv[1]
    if not os.path.isfile(svg_file):
        print(f"✖ File not found: {svg_file}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) >= 3:
        png_file = sys.argv[2]
    else:
        png_file = os.path.splitext(svg_file)[0] + ".png"

    convert(svg_file, png_file)

if __name__ == "__main__":
    main()
