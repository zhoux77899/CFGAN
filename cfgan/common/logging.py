import logging

FORMAT = "%(asctime)s [%(levelname)s] - %(filename)s (%(lineno)s): %(message)s"


logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
