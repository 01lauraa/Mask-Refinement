# Mask Refinement Pipeline

A modular Python pipeline for refining segmentation masks from vehicle damage inspection models. Raw predictions often contain artifacts — internal holes, disconnected noise, and jagged edges — that make them unsuitable for downstream business logic. This pipeline applies a configurable sequence of geometric operations to clean them up.

## Project Structure

```
├── src/
│   ├── operations.py       # Refinement operations and pipeline runner
│   ├── pipeline_config.py  # Configuration dataclass
│   └── pipeline_utils.py   # Orchestration, I/O, visualisation
├── scripts/
│   └── npz_mask_visualizer.py  # Standalone mask visualiser
├── tests/
│   └── test_mask_ops.py
├── data/
│   ├── masks/              # Input .npz files
│   └── output/             # Refined masks, config, comparison figure
├── main.py
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Output is written to `data/output/`:
- `mask_N_refined.npz` — refined masks ready for downstream use
- `comparison.png` — visual comparison of raw vs refined (only for batches ≤ 10 files)
- `config.json` — record of which operations were applied

To configure the pipeline, edit `main.py`:

```python
config = PipelineConfig(
    masks_dir="data/masks",
    output_dir="data/output",
    parallel=True,
    pipeline=[
        apply_hole_filling,
        constrain_to_main_foreground,
        partial(fill_gaps_nearest_neighbour, max_gap_area=250),
        partial(smooth_semantic_map, ksize=3),
    ],
)
```

## Architecture

### Plug-and-play pipeline

Each refinement operation is a standalone function with the signature `(masks: np.ndarray) -> np.ndarray`. The pipeline runner in `refine_masks` is a generic loop — it has no knowledge of individual operations:

```python
for step in pipeline:
    refined = step(refined)
```

Adding a new operation requires writing one function and appending it to the list in `main.py`. The runner never changes.

Parameters are bound to operations using `functools.partial`, keeping each step's signature consistent:

```python
partial(fill_gaps_nearest_neighbour, max_gap_area=500)
```

### Batch performance

In production, files are processed in parallel using `ProcessPoolExecutor`. Each file is independent, so refinement maps directly onto multiple CPU cores:

```python
with ProcessPoolExecutor() as executor:
    results = executor.map(_load_and_refine, args)
```

`_load_and_refine` is a top-level function so it can be pickled and sent to worker processes. Setting `parallel=False` in `PipelineConfig` disables this for local development, where process spawning overhead outweighs the benefit on small batches.

## Refinement Operations

| Operation | Addresses |
|---|---|
| `apply_hole_filling` | Internal holes caused by reflections or occlusions |
| `constrain_to_main_foreground` | Disconnected noise outside the main vehicle region |
| `fill_gaps_nearest_neighbour` | Small background gaps between adjacent parts |
| `smooth_semantic_map` | Jagged edges along part boundaries |

## Running Tests

```bash
pytest tests/
```

## Scaling to 10x More Data

The parallel architecture already scales linearly with CPU cores — 10x more images takes roughly 10x/N_cores longer, not 10x longer. For further scaling:

- **Chunked processing**: process files in batches to bound memory usage, saving refined masks per chunk rather than holding all results in RAM simultaneously
- **Distributed processing**: replace `ProcessPoolExecutor` with a task queue (e.g. Celery, Ray) to distribute across multiple machines
- **Cloud storage**: swap `np.load`/`np.savez` for cloud blob storage reads/writes without changing the operation functions

## Adding New Operations

Write a function that takes and returns a `(C, H, W)` NumPy array:

```python
def my_new_operation(masks: np.ndarray) -> np.ndarray:
    # process masks
    return refined_masks
```

Add it to the pipeline in `main.py`:

```python
pipeline=[
    apply_hole_filling,
    my_new_operation,       # insert anywhere in the sequence
    constrain_to_main_foreground,
    ...
]
```

No other files need to change.
