from pathlib import Path
from dataclasses import dataclass, field

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion, including paths and source URLs.
    """
    root_dir: Path
    source_url: str
    local_source_file: Path
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

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    """
    Configuration for callbacks during model training, including tensorboard and checkpoint paths.
    """
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for model training, including paths, training data, and training parameters.
    """
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    artifacts: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for model evaluation, including model path, data, and evaluation parameters.
    """
    path_of_model: Path
    training_data: Path
    artifacts: Path
    figures: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
