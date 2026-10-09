from copy import deepcopy
import io
from PIL import Image
import numpy as np
import pytest
from lifesci.core import (image_metrics, load_image, review_flags, sha256_bytes,
                          validate_record, score_predictions)
from lifesci.demo import synthetic_field
from lifesci.samples import example_record


def test_metrics_on_synthetic_demo():
    img = synthetic_field()
    q = image_metrics(img)
    assert q['width'] == q['height'] == 448
    assert q['p05_p95_contrast'] > 0
    assert q['laplacian_variance'] > 0


def test_variance_reacts_to_defocus():
    baseline = image_metrics(synthetic_field('baseline'))
    blurry = image_metrics(synthetic_field('defocused'))
    assert blurry['laplacian_variance'] < baseline['laplacian_variance']


def test_review_flags_exposure_and_low_contrast():
    dark = image_metrics(Image.new('RGB', (30, 30), color=(3, 3, 3)))
    assert any('underexposure' in s for s in review_flags(dark))
    light = image_metrics(Image.new('RGB', (30, 30), color=(252, 252, 252)))
    assert any('overexposure' in s for s in review_flags(light))
    assert any('low contrast' in s for s in review_flags(dark))


def test_load_image_rejects_corrupt():
    with pytest.raises(ValueError):
        load_image(b'not-an-image')


def test_load_image_converts_and_hashes():
    out = io.BytesIO()
    Image.new('L', (32, 32), color=120).save(out, format='PNG')
    img = load_image(out.getvalue())
    assert img.mode == 'RGB'
    assert len(sha256_bytes(out.getvalue())) == 64


def test_example_record_valid():
    rec = example_record('0' * 64)
    assert validate_record(rec) == []


def test_missing_metadata_rejected():
    rec = example_record('0' * 64)
    rec.pop('rights')
    assert 'Missing field: rights' in validate_record(rec)


def test_duplicate_answer_rejected():
    rec = example_record('0' * 64)
    rec['choices']['A'] = rec['choices']['B']
    assert any('distinct' in e for e in validate_record(rec))


def test_exact_match_evaluation():
    records = [{'id':'S1','answer':'A'}, {'id':'S2','answer':'C'}]
    report = score_predictions(records, [{'id':'S1','prediction':'a'}, {'id':'S2','prediction':'B'}])
    assert report['correct'] == 1
    assert report['accuracy'] == 0.5
    with pytest.raises(ValueError):
        score_predictions(records, [{'id':'UNKNOWN','prediction':'A'}])
