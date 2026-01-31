import importlib.metadata
import logging


__version__ = importlib.metadata.version("cs336_basics")

# Configure logging format with human-readable timestamps if not already configured
LOGGER = logging.getLogger("gpt_from_scratch")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)