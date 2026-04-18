from functools import partial
from src.pipeline_config import PipelineConfig
from src.pipeline_utils import run_pipeline
from src.operations import (
    apply_hole_filling,
    constrain_to_main_foreground,
    fill_gaps_nearest_neighbour,
)

if __name__ == "__main__":
    config = PipelineConfig(
        parallel=True,
        save_figure=True,
        pipeline=[
            #apply_hole_filling,
            constrain_to_main_foreground,
            partial(fill_gaps_nearest_neighbour, max_gap_area=250),
        ],
    )
    run_pipeline(config)
