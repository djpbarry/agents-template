from dataclasses import dataclass
from typing import Callable


@dataclass
class PipelineConfig:
    """Configuration for the multi-agent code-generation pipeline.

    This is the swap point for adapting the pipeline to a new domain.
    Create a new config file (e.g. bioimage_config.py) and pass an instance
    of PipelineConfig to generate_and_optimize().
    """
    orchestrator_model: str
    worker_model: str
    compiler_model: str
    evaluator_model: str
    docker_image: str
    available_libraries: str
    domain_notes: str
    extract_input_metadata: Callable[[str], str]
