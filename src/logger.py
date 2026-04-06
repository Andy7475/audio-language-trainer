# logger.py
import logging
import sys
import io


def setup_logger():
    logger = logging.getLogger("audio-language-trainer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File Handler - Always safe with utf-8 encoding
    file_handler = logging.FileHandler("app.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Console/Stream Handler
    # Check if we are in a standard terminal (has .buffer) or Jupyter (doesn't)
    if hasattr(sys.stdout, "buffer"):
        # Standard terminal: wrap for UTF-8 safety
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        console_handler = logging.StreamHandler(utf8_stdout)
    else:
        # Jupyter/IPython: sys.stdout already handles unicode fine
        console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create a single logger instance
logger = setup_logger()
