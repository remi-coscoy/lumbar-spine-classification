import os
import zipfile

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

from config.config import config
from custom_logging.logger import logger

try:
    data_config = config["data"]
    load_dotenv()

    # Set the dataset download path from the config
    dataset_path = data_config["path"]

    # Create the dataset directory if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)
    # Download dataset

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(
        competition=data_config["competition"],
        path=dataset_path,
    )
    # Unzip the downloaded file
    zip_file = os.path.join(dataset_path, data_config["competition"] + ".zip")
    if os.path.exists(zip_file):
        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(dataset_path)

        # Optionally remove the zip file after extraction
        os.remove(zip_file)
        logger.info(f"Dataset successfully downloaded and extracted to {dataset_path}")
    else:
        logger.info(
            "Download appears to have failed. Please check your internet connection and try again."
        )

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    logger.error("\nPossible solutions:")
    logger.error("Ensure you have set up your Kaggle API credentials:")
    logger.error(
        "   - Download your Kaggle API token from your Kaggle account settings"
    )
    logger.error(
        "   - Place the kaggle.json file in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<Windows-username>\\.kaggle\\"
    )
    logger.error(
        "3. Verify you have accepted the competition rules on the Kaggle website"
    )
    logger.error("4. Check if you have sufficient disk space")
