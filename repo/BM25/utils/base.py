import os
import logging

def line_count(filename: str) -> int:
    """Counts the number of lines in a file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())
    
def setup_logger(log_dir: str, log_filename: str) -> logging.Logger:
    """Sets up a logger that writes to a file in the specified directory."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s] - %(message)s')

    # Set Stream handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Set File handler
    fh = logging.FileHandler(os.path.join(log_dir, log_filename))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger