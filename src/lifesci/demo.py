"""Procedural images for UI smoke tests, NOT authentic microscopy."""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def synthetic_field(kind: str = "baseline", size: int = 448) -> Image.Image:
    """Return a labeled synthetic cell-like texture, not diagnostic biology."""
    if kind not in {"baseline", "defocused", "low_contrast", "overexposed"}:
        raise ValueError("Unknown demonstration type")
    rng = np.random.default_rng(12)
    bg = np.full((size, size, 3), [239, 213, 227], dtype=np.uint8)
    image = Image.fromarray(bg, "RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(57):
        x, y = (int(v) for v in rng.integers(15, size - 15, size=2))
        radius = int(rng.integers(8, 23))
        draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                     fill=(169, 86, 147, 120), outline=(90, 48, 140, 150), width=2)
        draw.ellipse([x-radius//3, y-radius//3, x+radius//3, y+radius//3],
                     fill=(71, 29, 112, 155))
    if kind == "defocused":
        image = image.filter(ImageFilter.GaussianBlur(5))
    if kind == "low_contrast":
        original = np.asarray(image, dtype=np.float32)
        image = Image.fromarray(np.uint8(np.clip(0.10 * original + 0.90 * 225, 0, 255)))
    if kind == "overexposed":
        original = np.asarray(image, dtype=np.float32)
        image = Image.fromarray(np.uint8(np.clip(original * 0.28 + 184, 0, 255)))
    return image
