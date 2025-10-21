import logging

def setup_logger(logger_name="peptron_logger", log_file="peptron.log"):
    """
    Set up and return a logger with both file and console handlers.
    
    Args:
        logger_name (str): Name of the logger
        log_file (str): Path to the log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Create a default logger instance
logger = setup_logger()

def inspect_arguments(self, *args, **kwargs):
    """
    Prints a detailed breakdown of all arguments passed to a function.
    
    Args:
        self: The instance of the class (if method)
        *args: Positional arguments
        **kwargs: Keyword arguments
    """
    logger.info("\n=== Argument Inspection ===")
    
    # Inspect self
    logger.info("\n🔵 Self:")
    logger.info(f"    Type: {type(self).__name__}")
    logger.info(f"    Value: {self}")
    
    # Inspect *args
    logger.info("\n🔵 Positional Arguments (*args):")
    if not args:
        logger.info("    No positional arguments")
    else:
        for i, arg in enumerate(args):
            logger.info(f"    {i}. Type: {type(arg).__name__}")
            logger.info(f"       Value: {arg}")
    
    # Inspect **kwargs
    logger.info("\n🔵 Keyword Arguments (**kwargs):")
    if not kwargs:
        logger.info("    No keyword arguments")
    else:
        for key, value in kwargs.items():
            logger.info(f"    {key}:")
            logger.info(f"        Type: {type(value).__name__}")
            logger.info(f"        Value: {value}")