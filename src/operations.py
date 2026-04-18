import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

# utility functions

def collapse_masks(masks: np.ndarray) -> np.ndarray:
    semantic = np.zeros(masks.shape[1:], dtype=masks.dtype)

    for channel in masks:
        semantic[channel > 0] = channel[channel > 0]

    return semantic

def expand_semantic_map(semantic: np.ndarray, reference_masks: np.ndarray) -> np.ndarray:
    class_ids = reference_masks.max(axis=(1, 2))
    matches = semantic[None] == class_ids[:, None, None]
    return (matches * class_ids[:, None, None]).astype(reference_masks.dtype)

# operations

def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes inside a binary mask.
    """
    binary_mask = binary_mask.astype(np.uint8)

    h, w = binary_mask.shape
    flood = binary_mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=1)

    holes = ((flood == 0) & (binary_mask == 0)).astype(np.uint8)
    return np.maximum(binary_mask, holes)

def apply_hole_filling(masks) -> np.ndarray:
    binary_masks = (masks > 0).astype(np.uint8)
    binary_masks = np.array([fill_holes(binary_masks[c]) for c in range(masks.shape[0])])
    class_ids = np.array([int(masks[c].max()) for c in range(masks.shape[0])])
    refined = binary_masks * class_ids[:, None, None]
    return refined

def constrain_to_main_foreground(masks: np.ndarray) -> np.ndarray:
    """
    keep only the largest connected foreground region across all channels.
    """
    combined_foreground = np.any(masks > 0, axis=0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined_foreground,
        connectivity=8,
    )

    if num_labels <= 1:
        return masks.copy()

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_region = labels == largest_label

    return masks * main_region[None, :, :]

def fill_gaps_nearest_neighbour(masks: np.ndarray, max_gap_area: int = 500) -> np.ndarray:
    """
    Fill small background gaps with the label of the nearest foreground pixel.
    """
    semantic = collapse_masks(masks)

    background = (semantic == 0).astype(np.uint8)
    if not background.any():
        return masks.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        background,
        connectivity=8,
    )

    small_gaps = np.zeros_like(background, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= max_gap_area:
            small_gaps[labels == i] = True

    if not small_gaps.any():
        return masks.copy()

    _, indices = distance_transform_edt(semantic == 0, return_indices=True)

    filled_semantic = semantic.copy()
    nearest_labels = semantic[indices[0], indices[1]]
    filled_semantic[small_gaps] = nearest_labels[small_gaps]

    return expand_semantic_map(filled_semantic, masks)

# operations pipeline

def refine_masks(masks: np.ndarray, pipeline: list) -> np.ndarray:
    if not isinstance(masks, np.ndarray):
        raise TypeError("masks must be a NumPy array")
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (C, H, W), got {masks.shape}")

    refined = masks.copy()
    for step in pipeline:
        refined = step(refined)
    return refined

