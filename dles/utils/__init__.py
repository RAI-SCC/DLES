import logging

import colorlog


def set_logger(level: int = logging.INFO, colors: bool = True, ) -> None:
    """
    Set configuration for the logger.

    Parameters
    __________
    level : int
        Logging level.
    colors : boolean
        Flag for using colors.
    """
    logger = logging.getLogger("dles")
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('log_file.log')
    if colors:
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)s: %(message)s',
            log_colors={
                'DEBUG': 'blue',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    logger.setLevel(level)
    logger.info(f'Logging level is {level}')
