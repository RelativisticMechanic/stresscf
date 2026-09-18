import sys
import logging

def getLogger(name):
    logger = logging.getLogger(name)
    
    # Prevents duplicate logs if this function is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(asctime)s %(name)s]: %(message)s', '%H:%M:%S'))
        
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger