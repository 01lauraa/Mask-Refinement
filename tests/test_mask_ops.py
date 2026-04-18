import sys
from pathlib import Path
from functools import partial
import numpy as np
import pytest
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.operations import (
    refine_masks,
    apply_hole_filling,
    constrain_to_main_foreground,
    fill_gaps_nearest_neighbour,
    smooth_semantic_map,
)

def test_invalid_shape_raises_value_error():
    """
    Edge case: input is not (C, H, W) → should raise ValueError.
    """
    masks = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        refine_masks(masks, [])

def test_empty_mask_returns_empty():
    """
    Edge case: completely empty mask should remain unchanged.
    """
    masks = np.zeros((3, 10, 10), dtype=np.uint8)

    pipeline = [
        apply_hole_filling,
        constrain_to_main_foreground,
        partial(fill_gaps_nearest_neighbour, max_gap_area=10),
        partial(smooth_semantic_map, ksize=3),
    ]

    refined = refine_masks(masks, pipeline)

    assert refined.shape == masks.shape
    assert np.array_equal(refined, masks)


def test_disconnected_object_removed():
    """
    Edge case: small disconnected region should be removed.
    """
    masks = np.zeros((1, 10, 10), dtype=np.uint8)

    # Main component
    masks[0, 1:5, 1:5] = 9

    # Detached smaller component
    masks[0, 8:10, 8:10] = 9

    refined = refine_masks(masks, [constrain_to_main_foreground])

    # Main region still exists
    assert np.any(refined[0, 1:5, 1:5] == 9)

    # Detached region removed
    assert np.all(refined[0, 8:10, 8:10] == 0)
