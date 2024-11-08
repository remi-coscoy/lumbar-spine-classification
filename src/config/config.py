import os
import yaml
from custom_logging.logger import logger

# Default configuration file
CONFIG_FILE = "config.yaml"
SAMPLE_CONFIG_FILE = "config-sample.yaml"


def load_config(config_file=CONFIG_FILE):
    """
    Load the configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file. Defaults to 'config.yaml'.

    Returns:
        dict: The loaded configuration.
    """
    config_path = os.path.join(os.path.dirname(__file__), config_file)

    if not os.path.exists(config_path):
        logger.warning(
            f"Configuration file '{config_file}' not found. Trying '{SAMPLE_CONFIG_FILE}'. Please copy the sample configuration and rename it to '{config_file}' to make changes"
        )
        config_path = os.path.join(os.path.dirname(__file__), SAMPLE_CONFIG_FILE)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


# Load the default configuration
config = load_config()
