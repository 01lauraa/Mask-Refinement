# Mask Refinement Pipeline

Pipeline for refining segmentation masks of vehicle parts. 

## Project Structure

```
├── src/
│   ├── operations.py       # Refinement operations and pipeline runner
│   ├── pipeline_config.py  # Configuration dataclass
│   └── pipeline_utils.py   # Orchestration, I/O, visualisation
├── scripts/
│   └── npz_mask_visualizer.py  # Mask visualiser
├── tests/
│   └── test_mask_ops.py
├── data/
│   ├── masks/              # Input masks
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

`masks/mask_N_refined.npz`: refined masks

`comparison.png`: visualize raw vs refined masks

`config.json`: record of which operations were applied

To configure the pipeline, edit `main.py`. 

```python
config = PipelineConfig(
        parallel=True,
        save_figure=True,
        pipeline=[
            #apply_hole_filling,
            constrain_to_main_foreground,
            partial(fill_gaps_nearest_neighbour, max_gap_area=250),
        ],
    )
```

## Architecture

Each refinement operation is a function that takes a mask as input and outputs the refined version. The function run_pipeline loops over a list of operations and applies them in sequence.

To add a new operation to the pipeline, a corresponding function that takes and returns a `(C, H, W)` NumPy array has to be defined in operations.py, and added to the list in `main.py`:

```python
pipeline=[
    apply_hole_filling,
    my_new_operation,
    constrain_to_main_foreground,
    ...
]
```

### Batch performance

When possible, operations avoid using loops and use NumPy broadcasting instead. 

To optimize for speed when processing larger batches of masks, files can be processed in parallel using `ProcessPoolExecutor` by setting `parallel=True`.

To optimize further, batch processing could be implemented to process files in groups and limit RAM usage.

## Refinement Operations

The following operations were tested in the pipeline:

`apply_hole_filling`: To fill out internal holes 

`constrain_to_main_foreground`: To remove false positives detected outside the car outline 

`fill_gaps_nearest_neighbour`: To fill out holes and small background gaps between adjacent masks 

## Possible additional post-processing steps
 
 Mask 2 seems to misclassify pixels at the bottom of the car as a mirror. To improve classification robustness, spatial filters could be added as a post-processing step to flag or suppress predictions where a part class appears outside its expected spatial region. Eg. rejecting a mirror prediction that falls below the lower half of the vehicle. 


## Running Tests

```bash
pytest tests/
```
Tests cover: 

`test_invalid_shape`: invalid input shape raises Error

`test_empty_mask`: empty  masks remains unchanged

`test_disconnected_object`: disconnected objects are removed

`test_hole_filling`: internsal holes are filled with background pixel ids

`test_gap_filling`: small gaps should be filled leaving background intact



