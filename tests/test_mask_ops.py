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
    Input is not (C, H, W) → should raise ValueError.
    """
    masks = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        refine_masks(masks, [])

def test_empty_mask_returns_empty():
    """
    empty mask should remain unchanged.
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
    Small disconnected region should be removed.
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


def test_hole_filling_fills_internal_hole():
    """
    Hole filling: internal background hole should be filled with the surrounding class id.
    """
    masks = np.zeros((1, 10, 10), dtype=np.uint8)

    # Solid ring with a hole in the middle
    masks[0, 2:8, 2:8] = 5
    masks[0, 4:6, 4:6] = 0

    refined = refine_masks(masks, [apply_hole_filling])

    # Hole should now be filled
    assert np.all(refined[0, 4:6, 4:6] == 5)

    # Surrounding region unchanged
    assert np.all(refined[0, 2:8, 2:8] == 5)


def test_gap_filling_preserves_large_background():
    """
    Gap filling: enclosed small gaps should be filled, large outer background should be untouched.
    """
    masks = np.zeros((1, 20, 20), dtype=np.uint8)

    # Solid foreground square with a small enclosed hole in the middle
    masks[0, 3:17, 3:17] = 3
    masks[0, 8:12, 8:12] = 0  # 4x4 = 16 pixel enclosed gap

    refined = refine_masks(masks, [partial(fill_gaps_nearest_neighbour, max_gap_area=20)])

    # Enclosed small gap should be filled
    assert np.all(refined[0, 8:12, 8:12] == 3)

    # Large outer background should remain untouched
    assert np.all(refined[0, 0:2, :] == 0)
    assert np.all(refined[0, 18:, :] == 0)
