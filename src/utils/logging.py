import logging
import warnings

logging.basicConfig(handlers=[logging.FileHandler("../logs/LOG.log"), logging.StreamHandler()],
                    format='%(asctime)s: %(levelname)s: %(message)s')

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_logger():
    """
    A simple function to prepare the logger.

    :return: a logger.
    :rtype: logging.Logger
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger
