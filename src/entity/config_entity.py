from src.constants import *
import os
from pathlib import Path
from dataclasses import dataclass
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                PrepareCallbacksConfig,
                                                TrainingConfig,
                                                EvaluationConfig)

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion, including paths and source URLs.
    """
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """
    Configuration for preparing the base model, including paths and model parameters.
    """
        root_dir: Path
        base_model_path: Path
        updated_base_model_path: Path
        params_image_size: list
        params_learning_rate: float
        params_include_top: bool
        params_weights: str
        params_classes: int
        params_freeze_all: bool = field(default=True)
        params_freeze_till: int = field(default=2)
