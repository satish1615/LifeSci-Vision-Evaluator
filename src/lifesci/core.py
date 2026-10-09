"""Deterministic image quality measurements and scientific annotation checks.

These checks are NOT medical interpretation or biological ground-truth validation.
They provide review prompts, not classifications of actual lab artifacts.
"""
from __future__ import annotations

import hashlib
import io
from typing import Any
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MODALITIES = (
    "Histopathology", "Cytology", "Hematology", "Molecular Imaging",
    "Agricultural Microbiology", "Other Life Sciences",
)
ARTIFACTS = (
    "Out-of-focus region", "Uneven illumination", "Stain variability",
    "Dust or debris", "Possible contamination", "Imaging noise",
    "Compression artifacts", "Motion artifact", "Overexposure", "Underexposure",
    "No visible artifact identified", "Uncertain / requires expert review",
)
LETTERS = ("A", "B", "C", "D")
MAX_PIXELS = 24_000_000


def load_image(payload: bytes) -> Image.Image:
    """Decode PNG/JPEG/TIFF safely, stripping EXIF orientation ambiguity."""
    if not payload or len(payload) > 25_000_000:
        raise ValueError("Choose an image smaller than 25 MB")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 24 megapixel limit")
            if opened.format not in {"PNG", "JPEG", "TIFF"}:
                raise ValueError("Supported formats: PNG, JPEG, TIFF")
            image = ImageOps.exif_transpose(opened)
            return image.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Unable to decode a supported image") from exc


def image_metrics(image: Image.Image) -> dict[str, float | int]:
    """Compute transparent pixel-level metrics, with no trained ML model."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected a 3-channel image")
    # ITU-R BT.709 luma weighting, on display-referred sRGB values (proxy).
    gray = np.dot(rgb, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    # Four-neighbor discrete Laplacian variance. High values can reflect noise,
    # texture or sharp edges; do not equate this with biological image quality.
    if min(gray.shape) >= 3:
        center = gray[1:-1, 1:-1]
        lap = 4 * center - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
        edge_variance = float(np.var(lap))
    else:
        edge_variance = 0.0
    return {
        "width": int(image.width),
        "height": int(image.height),
        "mean_luminance": round(float(np.mean(gray)), 4),
        "p05_p95_contrast": round(contrast, 4),
        "laplacian_variance": round(edge_variance, 6),
        "dark_pixel_fraction": round(float(np.mean(gray < 0.10)), 4),
        "bright_pixel_fraction": round(float(np.mean(gray > 0.90)), 4),
    }


def review_flags(metrics: dict, *, contrast_floor: float = 0.15,
                 edge_floor: float = 0.003) -> list[str]:
    """Return *candidate review prompts*, not artifact diagnoses."""
    flags = []
    if metrics["mean_luminance"] < 0.20:
        flags.append("Low overall brightness: inspect potential underexposure")
    if metrics["mean_luminance"] > 0.85:
        flags.append("High overall brightness: inspect potential overexposure")
    if metrics["p05_p95_contrast"] < contrast_floor:
        flags.append("Narrow tonal range: inspect low contrast or low signal")
    if metrics["laplacian_variance"] < edge_floor:
        flags.append("Low edge variation: inspect focus or low-texture field")
    if not flags:
        flags.append("No simple numeric threshold was triggered; expert review remains required")
    return flags


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_record(record: Any) -> list[str]:
    """Check the structure of a manually annotated VQA + image-description record."""
    if not isinstance(record, dict):
        return ["Record must be a JSON object"]
    required = (
        "id", "modality", "image_sha256", "provenance", "rights", "sample_type",
        "image_description", "quality_observations", "visible_artifacts",
        "question", "choices", "answer", "reasoning", "image_evidence",
        "review_status", "source_type",
    )
    errors = [f"Missing field: {name}" for name in required if name not in record]
    if errors:
        return errors
    for name in ("id", "provenance", "rights", "sample_type", "quality_observations",
                 "question", "reasoning", "image_evidence"):
        if not isinstance(record[name], str) or not record[name].strip():
            errors.append(f"{name} must be nonempty text")
    if record["modality"] not in MODALITIES:
        errors.append("Invalid modality")
    if not isinstance(record["image_sha256"], str) or len(record["image_sha256"]) != 64 or any(c not in '0123456789abcdef' for c in record["image_sha256"]):
        errors.append("image_sha256 must be a lowercase 64-character SHA-256")
    desc = record["image_description"]
    if not isinstance(desc, str) or len(desc.split()) < 35:
        errors.append("image_description must have at least 35 words, covering modality, quality, and visible features")
    options = record["choices"]
    if not isinstance(options, dict) or set(options) != set(LETTERS):
        errors.append("choices must have keys A, B, C, D")
    elif any(not isinstance(v, str) or not v.strip() for v in options.values()):
        errors.append("All answer choices must be filled")
    elif len({v.strip().casefold() for v in options.values()}) != 4:
        errors.append("Answer choices must all be distinct")
    if record["answer"] not in LETTERS:
        errors.append("answer must be A, B, C or D")
    artifacts = record["visible_artifacts"]
    if not isinstance(artifacts, list) or any(item not in ARTIFACTS for item in artifacts):
        errors.append("visible_artifacts must be a list of allowed reviewer labels")
    if record["review_status"] not in {"Draft", "Human-reviewed"}:
        errors.append("review_status must be Draft or Human-reviewed")
    if record["source_type"] not in {"Synthetic demonstration", "Licensed research image", "Own research image"}:
        errors.append("source_type must identify synthetic, licensed, or owned imagery")
    return errors


def score_predictions(records: list[dict], predictions: list[dict]) -> dict:
    """Exact-match scoring keyed by task ID; missing answers count as incorrect."""
    keys = {r['id']: r['answer'] for r in records}
    guesses = {}
    for row in predictions:
        if row.get('id') in guesses:
            raise ValueError(f"Duplicate prediction for {row['id']}")
        guesses[row.get('id')] = row.get('prediction', '').strip().upper()
    invalid_ids = set(guesses) - set(keys)
    if invalid_ids:
        raise ValueError(f"Unknown IDs: {', '.join(sorted(str(s) for s in invalid_ids))}")
    details = [{'id': k, 'gold': gold, 'prediction': guesses.get(k, ''),
                'correct': guesses.get(k) == gold} for k, gold in keys.items()]
    return {'total': len(details), 'correct': sum(x['correct'] for x in details),
            'accuracy': round(sum(x['correct'] for x in details) / len(details), 4) if details else 0,
            'details': details}
