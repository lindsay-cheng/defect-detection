"""logging setup for entry points"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """call once per entry point"""
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
