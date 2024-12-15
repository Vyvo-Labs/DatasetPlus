import logging
import colorlog

def setup_logger(name: str = "datasetplus") -> logging.Logger:
    """Set up a colored logger instance.
    
    Args:
        name (str, optional): Name of the logger. Defaults to "datasetplus".
        
    Returns:
        logging.Logger: Configured logger instance with colored output.
    """
    logger = colorlog.getLogger(name)
    
    if logger.handlers:
        return logger
        
    # Create console handler with formatting
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )
    
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    return logger

# Create default logger instance
logger = setup_logger()
