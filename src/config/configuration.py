from src.constants import *
import os
from pathlib import Path
from src.utils.auxiliary_functions import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                PrepareCallbacksConfig,
                                                TrainingConfig,
                                                EvaluationConfig)


class ConfigurationManager:
    """
    This class is used to manage the configuration settings for various stages of a machine learning pipeline.

    Attrs:
        config (dict): A dictionary containing configuration settings.
        params (dict): A dictionary containing parameter settings.
    """

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        """
        Initializes the ConfigurationManager with configuration and parameter files.

        Args:
            config_filepath (str): The file path for the configuration settings. Defaults to CONFIG_FILE_PATH.
            params_filepath (str): The file path for the parameter settings. Defaults to PARAMS_FILE_PATH.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config['artifacts_root']])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        This function retrieves the data ingestion configuration.

        return:
            DataIngestionConfig: An object containing the data ingestion configuration.
        """
        config = self.config['data_ingestion']

        create_directories([config['root_dir']])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config['root_dir'],
            source_url=config['source_url'],
            local_source_file=config.get('local_source_file'),
            local_data_file=config['local_data_file'],
            unzip_dir=config['unzip_dir']
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        This function retrieves the base model preparation configuration.

        return:
            PrepareBaseModelConfig: An object containing the base model preparation configuration.
        """
        config = self.config['prepare_base_model']

        create_directories([config['root_dir']])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config['root_dir']),
            base_model_path=Path(config['base_model_path']),
            updated_base_model_path=Path(config['updated_base_model_path']),
            params_image_size=self.params['IMAGE_SIZE'],
            params_learning_rate=self.params['LEARNING_RATE'],
            params_include_top=self.params['INCLUDE_TOP'],
            params_weights=self.params['WEIGHTS'],
            params_classes=self.params['CLASSES'],
            params_freeze_all=self.params['FREEZE_ALL'],
            params_freeze_till=self.params.get('FREEZE_TILL', None)
        )

        return prepare_base_model_config

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        """
        This function retrieves the callback preparation configuration.

        return:
            PrepareCallbacksConfig: An object containing the callback preparation configuration.
        """
        config = self.config['prepare_callbacks']
        model_ckpt_dir = os.path.dirname(config['checkpoint_model_filepath'])
        create_directories([
            Path(model_ckpt_dir),
            Path(config['tensorboard_root_log_dir'])
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config['root_dir']),
            tensorboard_root_log_dir=Path(config['tensorboard_root_log_dir']),
            checkpoint_model_filepath=Path(config['checkpoint_model_filepath'])
        )

        return prepare_callback_config

    def get_training_config(self) -> TrainingConfig:
        """
        This function retrieves the training configuration.

        return:
            TrainingConfig: An object containing the training configuration.
        """
        training = self.config['training']
        prepare_base_model = self.config['prepare_base_model']
        params = self.params
        training_data = os.path.join(self.config['data_ingestion']['unzip_dir'], "Chicken-fecal-images")
        create_directories([
            Path(training['root_dir'])
        ])

        training_config = TrainingConfig(
            root_dir=Path(training['root_dir']),
            trained_model_path=Path(training['trained_model_path']),
            updated_base_model_path=Path(prepare_base_model['updated_base_model_path']),
            training_data=Path(training_data),
            params_epochs=params['EPOCHS'],
            params_batch_size=params['BATCH_SIZE'],
            params_is_augmentation=params['AUGMENTATION'],
            params_image_size=params['IMAGE_SIZE']
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        """
        This function retrieves the evaluation configuration.

        return:
            EvaluationConfig: An object containing the evaluation configuration.
        """
        eval_config = self.config['evaluation']
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/Chicken-fecal-images",
            mlflow_uri="TBD",
            all_params=self.params,
            params_image_size=self.params['IMAGE_SIZE'],
            params_batch_size=self.params['BATCH_SIZE']
        )
        return eval_config