import pytest
from PIL import Image
from lifesci.core import compare_image_quality
from lifesci.demo import synthetic_field


def test_compares_synthetic_image_conditions():
    rows = compare_image_quality([('baseline', synthetic_field('baseline')),
                                  ('defocused', synthetic_field('defocused'))])
    assert len(rows) == 2
    assert rows[0]['condition'] == 'baseline'
    assert rows[0]['laplacian_variance'] > rows[1]['laplacian_variance']
    assert rows[0]['review_prompts']


def test_duplicated_labels_rejected():
    with pytest.raises(ValueError, match='unique'):
        compare_image_quality([('one', Image.new('RGB', (16, 16))),
                               ('one', Image.new('RGB', (16, 16)))])
