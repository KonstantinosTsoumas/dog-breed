=======================
Project Pipeline Documentation
=======================

This documentation explains the pipeline scripts used in the project for various stages like data ingestion, model preparation, training, and evaluation.

Data Ingestion Pipeline (`data_ingestion_01.py`)
------------------------------------------------

This script handles the data ingestion process, including downloading a file and extracting a ZIP file.

.. code-block:: python

   from src.config.configuration import ConfigurationManager
   from src.components.data_ingestion import DataIngestion
   from src import logger

   STAGE_NAME = "Step 01: Data Ingestion"

   class DataIngestionTrainingPipeline:
       """
       This class represents the data ingestion pipeline.

       It is responsible for downloading a file and extracting a ZIP file as part of the data ingestion process.
       """

       def __init__(self):
           pass

       def main(self):
           """
           This function is responsible for running the data ingestion part step-by-step.
           """
           config = ConfigurationManager()
           data_ingestion_config = config.get_data_ingestion_config()
           data_ingestion = DataIngestion(config=data_ingestion_config)

           # Download the file
           data_ingestion.download_file()

           # Extract the ZIP file
           data_ingestion.extract_zip_file()

   if __name__ == '__main__':
       # Exception handling for DVC included
       try:
           logger.info(f"------ {STAGE_NAME} has started ------")
           obj = DataIngestionTrainingPipeline()
           obj.main()
           logger.info(f"------  {STAGE_NAME} has been completed ------")
       except Exception as e:
           logger.exception(e)
           raise e

Prepare Base Model Pipeline (`prepare_base_model_02.py`)
--------------------------------------------------------

This script prepares the base model for further use in the project.

.. code-block:: python

   from src.config.configuration import ConfigurationManager
   from src.components.prepare_base_model import PrepareBaseModel
   from src import logger

   STAGE_NAME = "Preparing base model started"

   class PrepareBaseModelTrainingPipeline:
       """
       This class represents the pipeline for preparing the base model.

       It obtains and updates the base model according to the specified configuration.
       """

       def __init__(self):
           pass

       def main(self):
           config = ConfigurationManager()
           prepare_base_model_config = config.get_prepare_base_model_config()
           prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
           prepare_base_model.get_base_model()
           prepare_base_model.update_base_model()

   if __name__ == '__main__':
       try:
           logger.info(f"------ {STAGE_NAME} started ------")
           obj = PrepareBaseModelTrainingPipeline()
           obj.main()
           logger.info(f"------ {STAGE_NAME} has been completed ------")
       except Exception as e:
           logger.exception(e)
           raise e


Model Training Pipeline (`model_training_03.py`)
------------------------------------------------

This script manages the model training process, including the setup of callbacks.

.. code-block:: python

   from src.config.configuration import ConfigurationManager
   from src.components.prepare_callbacks import PrepareCallback
   from src.components.model_training import Training
   from src import logger

   STAGE_NAME = "Training"

   class ModelTrainingPipeline:
       """
       This class represents the pipeline for model training.

       It sets up training callbacks, configures the training process, and initiates model training.
       """

       def __init__(self):
           pass

       def main(self):
           config = ConfigurationManager()
           prepare_callbacks_config = config.get_prepare_callback_config()
           prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
           callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
           training_config = config.get_training_config()
           training = Training(config=training_config)
           training.get_base_model()
           training.train(callback_list=callback_list)

   if __name__ == '__main__':
       try:
           logger.info(f"stage {STAGE_NAME} has started.")
           obj = ModelTrainingPipeline()
           obj.main()
           logger.info(f"stage {STAGE_NAME} has been completed.")
       except Exception as e:
           logger.exception(e)
           raise e

Evaluation Pipeline (`evaluation.py`)
--------------------------------------

This script handles the evaluation tasks, including scoring and plotting ROC curves.

.. code-block:: python

   from src.config.configuration import ConfigurationManager
   from src.components.evaluation import Evaluation
   from src import logger

   STAGE_NAME = "Evaluation stage"

   class EvaluationPipeline:
       """
       This class represents the pipeline for model evaluation.

       It computes scores, saves them, and plots top ROC curves based on the evaluation results.
       """

       def __init__(self):
           pass

       def main(self):
           config = ConfigurationManager()
           val_config = config.get_evaluation_config()
           evaluation = Evaluation(val_config)
           scores, y_true_binarized, predictions = evaluation.evaluation()
           evaluation.save_score(scores)
           evaluation.plot_top_roc_curves(scores, y_true_binarized, predictions)

   if __name__ == '__main__':
       try:
           logger.info(f"stage {STAGE_NAME} has started.")
           obj = EvaluationPipeline()
           obj.main()
           logger.info(f"stage {STAGE_NAME} has been completed.")
       except Exception as e:
           logger.exception(e)
           raise e

# Continue with documentation for other scripts...


.. note::
   Ensure that you have the required dependencies installed to run these pipeline scripts successfully. Refer to the project documentation for additional details on running and configuring the pipeline.

