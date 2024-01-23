import os
import urllib.request as request
import zipfile
from src import logger
from src.utils.auxiliary_functions import get_size
from src.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        This function initializes the DataIngestion instance.

        Parameters:
        config (DataIngestionConfig): Configuration object containing parameters for data ingestion.
        """
        self.config = config

    def download_file(self):
        """
        This function checks if the file needs to be downloaded from a URL or copied from a local path.
        """
        if not os.path.exists(self.config.local_data_file):
            if hasattr(self.config, 'local_source_file') and os.path.exists(self.config.local_source_file):
                # Copy file from local source
                shutil.copyfile(self.config.local_source_file, self.config.local_data_file)
                logger.info(f"Local file {self.config.local_source_file} copied to {self.config.local_data_file}")
            else:
                # Download from URL (existing logic)
                logger.info(f"Local source file not found or not specified, attempting to download from URL.")
                filename, headers = request.urlretrieve(
                    url=self.config.source_url,
                    filename=self.config.local_data_file
                )
                logger.info(f"{filename} has been downloaded! More info here: \n{headers}")
        else:
            logger.info(f"File already exists, size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        This function extracts the zip file into the specified directory. It also, creates the directory if it does not exist.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

