import logging

NOISY_LOGGERS = ("httpx", "requests", "urllib3")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
