import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Car Part Mapping
CLASS_MAP = {
    0: "BACKGROUND", 1: "FUEL_DOOR", 2: "WINDOW", 3: "SIDE_PANEL",
    4: "ROOF_RAILS", 5: "SPOILER", 6: "BUMPER", 7: "PILLAR_A",
    8: "MIRROR", 9: "DOOR", 10: "WHEEL", 11: "BONNET",
    12: "DOOR_HANDLE", 13: "SILL", 14: "ROOF", 15: "HEAD_LIGHT",
    16: "FENDER", 17: "TAIL_LIGHT", 18: "TRUNK"
}


def load_and_collapse_mask(npz_path):
    """
    Collapses multi-channel instance masks into a single 2D semantic mask.
    The pixel values inside each channel denote the class_id.
    Mask dimensions: num_instances x 284 X 340 (resized small)
    """
    with np.load(npz_path) as data:
        # Load the array from the first key in the npz archive
        masks = data[data.files[0]]

        # Initialize a 2D canvas (Height, Width)
    collapsed = np.zeros(masks.shape[1:], dtype=np.float32)

    # Flattening logic: Extract the class_id from each channel
    # and apply it to the 2D canvas.
    for channel in masks:
        class_id = np.max(channel)
        if class_id > 0:
            # Stamp the class_id onto the canvas where the instance exists
            collapsed[channel > 0] = class_id

    return collapsed


def visualize_mask(mask_2d):
    """Displays the 2D mask with a legend, treating 0 as transparent."""
    display_mask = mask_2d.copy()
    display_mask[display_mask == 0] = np.nan  # Hide background

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab20', 20)
    plt.imshow(display_mask, cmap=cmap, interpolation='nearest', vmin=0, vmax=19)

    # Legend for present parts only
    present_ids = np.unique(mask_2d[mask_2d > 0]).astype(int)
    legend_elements = [Patch(facecolor=cmap(i / 20), label=CLASS_MAP[i]) for i in present_ids]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    visualize_mask(load_and_collapse_mask("masks\mask_2.npz"))