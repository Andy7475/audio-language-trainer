# logger.py
import logging
import sys

def setup_logger():
    """Configure and return the application logger."""
    logger = logging.getLogger('audio-language-trainer')  # Use your project name
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers to avoid duplicates if called multiple times
    logger.handlers.clear()
    
    # File handler - overwrites each run
    file_handler = logging.FileHandler('app.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler (optional but useful)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create a single logger instance
logger = setup_logger()