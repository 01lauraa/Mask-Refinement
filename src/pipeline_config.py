from dataclasses import dataclass, field

@dataclass
class PipelineConfig:
    masks_dir: str = "data/masks"
    output_dir: str = "data/output"
    parallel: bool = True
    save_figure: bool = True
    pipeline: list = field(default_factory=list)
    
