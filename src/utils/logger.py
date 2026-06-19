import logging
import sys
from src.utils.config import settings

def get_logger(name):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        log_level_str = settings.logging.get("level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        log_file = settings.logging.get("log_file", "logs/perception.log")
        try:
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Warning: Could not set up file logger at {log_file}. {e}")

        logger.propagate = False

    return logger
