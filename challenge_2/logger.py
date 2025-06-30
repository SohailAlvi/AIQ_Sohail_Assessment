import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = "app.log"

def setup_logger(name=None, log_file=LOG_FILE, level=logging.INFO):
    """
    Setup a logger with rotating file handler and console handler.
    Avoids adding duplicate handlers.
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()  # Prevent duplicate logs

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(module)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_file), maxBytes=1_000_000, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
