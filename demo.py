import logging
from rich.logging import RichHandler

# 1. Configure the Rich Logger
# We keep the format minimal because Rich automatically adds
# the timestamp, log level, and file path in neat columns.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

logger = logging.getLogger("demo")


def run_logging_demo():
    # 2. Standard log levels get automatic color coding
    logger.info("Application started successfully.")
    logger.warning("Memory usage is getting high.")

    # 3. You can use BBCode-style markup directly in your logs
    logger.info("Processing [bold cyan]Season 1, Episode 4[/bold cyan]...")

    # 4. The killer feature: Rich Tracebacks
    try:
        # Forcing a deliberate math error to trigger an exception
        result = 1 / 0
    except ZeroDivisionError:
        logger.exception("A critical calculation failed during execution!")


if __name__ == "__main__":
    run_logging_demo()
