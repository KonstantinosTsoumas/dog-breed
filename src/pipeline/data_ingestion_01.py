from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src import logger

STAGE_NAME = "Step 01: Data Ingestion"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        This function is responsible for running the data ingestion part step-by-step.
        """
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
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