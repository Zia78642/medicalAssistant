import logging

def setup_logger(name="MedicalAssistant"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] --- [%(message)s]")
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(ch)

    return logger

logger = setup_logger()

logger.info("RAG process started")
logger.debug("Debugging")
logger.error("Critical message")

