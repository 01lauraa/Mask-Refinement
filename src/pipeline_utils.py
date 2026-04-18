from pathlib import Path
import json
import numpy as np
from src.pipeline_config import PipelineConfig
from concurrent.futures import ProcessPoolExecutor

from src.operations import refine_masks, collapse_masks


def _load_and_refine(args: tuple) -> tuple:
    """Load a single .npz file and run the refinement pipeline.
    Top-level function so it can be pickled by ProcessPoolExecutor."""
    npz_path, pipeline = args
    with np.load(npz_path) as data:
        raw_masks = data[data.files[0]]
    refined_masks = refine_masks(raw_masks, pipeline)
    return npz_path, raw_masks, refined_masks


def generate_display_masks(raw_masks: np.ndarray, refined_masks: np.ndarray) -> tuple:
    raw_collapsed = collapse_masks(raw_masks).astype(np.float32)
    refined_collapsed = collapse_masks(refined_masks).astype(np.float32)

    raw_display = raw_collapsed.copy()
    raw_display[raw_display == 0] = np.nan

    refined_display = refined_collapsed.copy()
    refined_display[refined_display == 0] = np.nan
    return raw_display, refined_display


def save_comparison_figure(results: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    npz_paths = list(results.keys())
    fig, axes = plt.subplots(len(npz_paths), 2, figsize=(10, 4 * len(npz_paths)))

    if len(npz_paths) == 1:
        axes = np.array([axes])

    cmap = plt.get_cmap("tab20", 20)

    for row, npz_path in enumerate(npz_paths):
        raw_masks, refined_masks = results[npz_path]
        raw_display, refined_display = generate_display_masks(raw_masks, refined_masks)

        axes[row, 0].imshow(raw_display, cmap=cmap, interpolation="nearest", vmin=0, vmax=19)
        axes[row, 0].set_title(f"{npz_path.name} — Raw")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(refined_display, cmap=cmap, interpolation="nearest", vmin=0, vmax=19)
        axes[row, 1].set_title(f"{npz_path.name} — Refined")
        axes[row, 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_config(config: PipelineConfig, output_dir: Path) -> None:
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "pipeline": [step.func.__name__ if hasattr(step, "func") else step.__name__ for step in config.pipeline]
        }, f)


def _save_refined_masks(results: dict, output_dir: Path) -> None:
    for npz_path, (_, refined_masks) in results.items():
        out_path = output_dir / f"{npz_path.stem}_refined.npz"
        np.savez(out_path, refined_masks)


def _process_files(npz_paths: list, pipeline: list, parallel: bool) -> list:
    args = [(p, pipeline) for p in npz_paths]
    if parallel:
        with ProcessPoolExecutor() as executor:
            return list(executor.map(_load_and_refine, args))
    return [_load_and_refine(a) for a in args]


def run_pipeline(config: PipelineConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(Path(config.masks_dir).glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {config.masks_dir}")

    _save_config(config, output_dir)
    raw_results = _process_files(npz_paths, config.pipeline, config.parallel)

    results = {path: (raw, refined) for path, raw, refined in raw_results}
    _save_refined_masks(results, output_dir)
    if config.save_figure:
        save_comparison_figure(results, output_dir / "comparison.png")
    return results
