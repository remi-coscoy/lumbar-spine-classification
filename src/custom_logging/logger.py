import logging
import os

from config.config import config

logging_config = config["logging"]
# Set the log level from the configuration
LOG_LEVEL = logging_config.get("log_level", "INFO")

# Create the logger
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s \ \n %(message)s")
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Add a file handler
if logging_config.get("log_file"):
    log_file_path = os.path.join(
        os.path.dirname(__file__), "..", "..", logging_config["log_file"]
    )
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Export the logger
__all__ = ["logger"]
