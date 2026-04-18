"""
Benchmark: Sequential vs ThreadPoolExecutor vs ProcessPoolExecutor
Simulates 200 files by copying existing masks.
"""
import sys
import shutil
import time
import tempfile
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.operations import refine_masks, apply_hole_filling, constrain_to_main_foreground, fill_gaps_nearest_neighbour, smooth_semantic_map
from src.pipeline_config import PipelineConfig
import numpy as np

N_FILES = 200

CONFIG = PipelineConfig(
    parallel=True,
    save_figure=False,
    pipeline=[
        apply_hole_filling,
        constrain_to_main_foreground,
        partial(fill_gaps_nearest_neighbour, max_gap_area=250),
        partial(smooth_semantic_map, ksize=3),
    ],
)


def _load_and_refine(args):
    npz_path, pipeline = args
    with np.load(npz_path) as data:
        masks = data[data.files[0]]
    refine_masks(masks, pipeline)
    return npz_path


def setup_sim_dir(source_dir: Path, n: int) -> Path:
    tmp = Path(tempfile.mkdtemp())
    source_files = sorted(source_dir.glob("*.npz"))
    for i in range(n):
        src = source_files[i % len(source_files)]
        shutil.copy(src, tmp / f"mask_{i:04d}.npz")
    return tmp


def run(label, executor_cls, npz_paths):
    args = [(p, CONFIG.pipeline) for p in npz_paths]
    start = time.perf_counter()
    if executor_cls is None:
        for a in args:
            _load_and_refine(a)
    else:
        with executor_cls() as ex:
            list(ex.map(_load_and_refine, args))
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.2f}s")


if __name__ == "__main__":
    masks_dir = Path(CONFIG.masks_dir)
    print(f"Simulating {N_FILES} files...\n")
    tmp_dir = setup_sim_dir(masks_dir, N_FILES)

    try:
        npz_paths = sorted(tmp_dir.glob("*.npz"))
        run("Sequential          ", None, npz_paths)
        run("ThreadPoolExecutor  ", ThreadPoolExecutor, npz_paths)
        run("ProcessPoolExecutor ", ProcessPoolExecutor, npz_paths)
    finally:
        shutil.rmtree(tmp_dir)
