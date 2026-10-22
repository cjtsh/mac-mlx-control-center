#!/usr/bin/env python3
"""Generate the MLX Controller app icon.

Dark rounded-square tile, a cyan -> violet power glyph, and a soft glow.
Rendered at 4x then downsampled so the 16 px version stays crisp.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


HERE = os.path.dirname(os.path.abspath(__file__))
S = 4096
FINAL = 1024
ICONSET = os.path.join(HERE, "AppIcon.iconset")
ICNS = os.path.join(HERE, "AppIcon.icns")


def lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gradient(size: int, a: tuple[int, int, int], b: tuple[int, int, int], diagonal: bool = True) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    t = (xx + yy) / (size * 2.0) if diagonal else yy / float(size)
    t = np.clip(t, 0.0, 1.0)[..., None]
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    rgb = av + (bv - av) * t
    return Image.fromarray(rgb.round().astype(np.uint8), "RGB")


def rounded_mask(size: int, inset: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([inset, inset, size - inset, size - inset], radius=radius, fill=255)
    return mask


def main() -> int:
    tile_inset = int(S * 0.0625)
    tile_radius = int(S * 0.225)
    mask = rounded_mask(S, tile_inset, tile_radius)

    # Base: deep navy gradient with a cool lift in the lower third.
    base = gradient(S, (9, 13, 27), (23, 30, 58)).convert("RGBA")

    # Soft cyan bloom behind the glyph.
    bloom = Image.new("L", (S, S), 0)
    bd = ImageDraw.Draw(bloom)
    cx = cy = S // 2
    for r, alpha in ((int(S * 0.44), 26), (int(S * 0.30), 34), (int(S * 0.18), 44)):
        bd.ellipse([cx - r, cy - int(r * 0.95), cx + r, cy + int(r * 0.95)], fill=alpha)
    bloom = bloom.filter(ImageFilter.GaussianBlur(int(S * 0.09)))
    bloom_color = Image.new("RGBA", (S, S), (56, 189, 248, 0))
    bloom_color.putalpha(bloom)
    base = Image.alpha_composite(base, bloom_color)

    # Faint orbital rings for depth.
    rings = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rings)
    for r in (int(S * 0.30), int(S * 0.38)):
        rd.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(125, 211, 252, 16), width=max(2, S // 1024))
    base = Image.alpha_composite(base, rings)

    # Glyph mask: power ring with a gap at the top + a vertical stem.
    glyph = Image.new("L", (S, S), 0)
    gd = ImageDraw.Draw(glyph)
    radius = int(S * 0.165)
    thickness = int(S * 0.052)
    box = [cx - radius, cy - radius, cx + radius, cy + radius]
    gd.arc(box, start=300, end=240, fill=255, width=thickness)
    for deg in (300, 240):
        cr = radius - thickness / 2
        ex = cx + cr * math.cos(math.radians(deg))
        ey = cy + cr * math.sin(math.radians(deg))
        gd.ellipse(
            [ex - thickness / 2, ey - thickness / 2, ex + thickness / 2, ey + thickness / 2],
            fill=255,
        )
    stem_top = cy - radius - int(thickness * 0.55)
    stem_bottom = cy + int(thickness * 0.10)
    stem_w = int(thickness * 0.82)
    gd.rounded_rectangle(
        [cx - stem_w // 2, stem_top, cx + stem_w // 2, stem_bottom],
        radius=stem_w // 2,
        fill=255,
    )

    # Glyph glow.
    glow_mask = glyph.filter(ImageFilter.GaussianBlur(int(thickness * 1.35)))
    glow_color = Image.new("RGBA", (S, S), (34, 211, 238, 0))
    glow_color.putalpha(glow_mask.point(lambda v: int(v * 0.62)))
    base = Image.alpha_composite(base, glow_color)

    # Glyph gradient: cyan at top-left, violet at bottom-right.
    glyph_grad = gradient(S, (94, 234, 255), (139, 92, 246)).convert("RGBA")
    glyph_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glyph_layer.paste(glyph_grad, (0, 0), glyph)
    base = Image.alpha_composite(base, glyph_layer)

    # Crisp inner rim light along the top edge.
    rim = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(rim)
    rdraw.rounded_rectangle(
        [tile_inset + 2, tile_inset + 2, S - tile_inset - 2, S - tile_inset - 2],
        radius=tile_radius - 2,
        outline=(255, 255, 255, 38),
        width=max(2, S // 850),
    )
    base = Image.alpha_composite(base, rim)

    # Apply the rounded tile mask, preserving transparency outside it.
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(base, (0, 0), mask)
    out = out.resize((FINAL, FINAL), Image.LANCZOS)
    out.save(os.path.join(HERE, "AppIcon-1024.png"))

    if os.path.isdir(ICONSET):
        shutil.rmtree(ICONSET)
    os.makedirs(ICONSET)
    specs = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]
    for size, name in specs:
        out.resize((size, size), Image.LANCZOS).save(os.path.join(ICONSET, name))
    subprocess.run(["iconutil", "-c", "icns", ICONSET, "-o", ICNS], check=True)
    print("wrote", ICNS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
