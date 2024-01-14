from src import logger
import tensorflow as tf
from src.pipeline.data_ingestion_01 import DataIngestionTrainingPipeline
from src.pipeline.prepare_base_model_02 import PrepareBaseModelTrainingPipeline
from src.pipeline.model_training_03 import ModelTrainingPipeline
# from src.pipeline.stage_04_evaluation import EvaluationPipeline


STAGE_NAME = "Data Ingestion phase"
try:
   logger.info(f" {STAGE_NAME} started !")
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f" {STAGE_NAME} has been completed.")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Prepare base model"
try:
   logger.info(f" {STAGE_NAME} started !")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f" {STAGE_NAME} has been completed.")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Training"
try:
   logger.info(f" {STAGE_NAME} started !")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f" {STAGE_NAME} has been completed.")
except Exception as e:
        logger.exception(e)
        raise e
