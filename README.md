# Mask Refinement Pipeline

A modular Python pipeline for refining segmentation masks from vehicle damage inspection models. Raw predictions often contain artifacts — internal holes, disconnected noise, and jagged edges — that make them unsuitable for downstream business logic.

## Project Structure

```
├── src/
│   ├── operations.py       # Refinement operations and pipeline runner
│   ├── pipeline_config.py  # Configuration dataclass
│   └── pipeline_utils.py   # Orchestration, I/O, visualisation
├── scripts/
│   ├── benchmark.py            # Sequential vs parallel performance comparison
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
- `masks/mask_N_refined.npz` — refined masks ready for downstream use
- `comparison.png` — side-by-side visual of raw vs refined masks
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

Each refinement operation is a standalone function with the signature `(masks: np.ndarray) -> np.ndarray`. The pipeline runner applies them in sequence:

```python
for step in pipeline:
    refined = step(refined)
```

The runner has no knowledge of individual operations — it just loops. Adding a new operation means writing one function and appending it to the list in `main.py`. No other files change.

Operation parameters are bound with `functools.partial`, so each step always presents the same interface to the runner:

```python
partial(fill_gaps_nearest_neighbour, max_gap_area=500)
```

### Batch performance

In production, files are processed in parallel using `ProcessPoolExecutor`. Each file is independent, so the work maps directly onto multiple CPU cores:

```python
with ProcessPoolExecutor() as executor:
    results = executor.map(_load_and_refine, args)
```

`_load_and_refine` is defined at module level so it can be pickled and sent to worker processes — a requirement for Python multiprocessing. Setting `parallel=False` in `PipelineConfig` disables this for local development, where the process spawn overhead outweighs the benefit on small batches.

## Refinement Operations

| Operation | Addresses |
|---|---|
| `apply_hole_filling` | Internal holes caused by reflections or occlusions |
| `constrain_to_main_foreground` | Disconnected noise outside the main vehicle region |
| `fill_gaps_nearest_neighbour` | Small background gaps between adjacent parts |

## Data Observations

Not all provided masks required intervention. Masks 1, 3, and 4 had relatively clean predictions. Mask 2 contained a notable misclassification: pixels at the bottom of the car body were labelled as *mirror*, a geometrically implausible prediction given the spatial position of that class.

The current pipeline does not correct semantic misclassifications — only geometric artifacts. This is a deliberate scope decision: geometric refinement is class-agnostic and generalises across all parts, whereas fixing semantic errors requires class-specific rules.

## Running Tests

```bash
pytest tests/
```

Tests cover: invalid input shape, empty masks, disconnected region removal, internal hole filling, and gap filling with large background preservation.

## Scaling to 10x More Data

The parallel architecture already scales with CPU cores — 10x more images takes roughly 10x/N_cores longer, not 10x longer. For further scaling:

- **Chunked processing**: process files in batches to bound memory usage, saving refined masks per chunk rather than holding all results in RAM simultaneously
- **Distributed processing**: replace `ProcessPoolExecutor` with a task queue (e.g. Celery, Ray) to distribute across multiple machines
- **Cloud storage**: swap `np.load`/`np.savez` for cloud blob storage reads/writes without changing any operation functions

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
    my_new_operation,
    constrain_to_main_foreground,
    ...
]
```

No other files need to change.

## Suggested Post-processing

For semantic misclassifications like the one observed in mask 2, a spatial consistency filter could be added as a post-processing step. This would flag or suppress predictions where a part class appears outside its expected spatial region — for example, rejecting a *mirror* prediction that falls below the vertical midpoint of the vehicle bounding box. Such a filter would integrate naturally into the existing pipeline as an additional callable step.
