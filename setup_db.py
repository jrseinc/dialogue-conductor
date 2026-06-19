import os
import time
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from rich.logging import RichHandler
from rich.console import Console

load_dotenv()
load_dotenv(".env.local", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("rich")
console = Console()


def ensure_index(pc: Pinecone, index_name: str, existing: list[str]) -> None:
    """Create the index if it doesn't already exist, then wait until it's ready."""
    if index_name in existing:
        logger.info(
            f"Index '[bold green]{index_name}[/bold green]' already exists.",
            extra={"markup": True},
        )
        return

    logger.info(
        f"Creating '[bold cyan]{index_name}[/bold cyan]' index...",
        extra={"markup": True},
    )

    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    with console.status(
        f"[bold yellow]Waiting for cloud index '{index_name}' to initialize...[/bold yellow]"
    ):
        while True:
            index_info = pc.describe_index(index_name)

            if index_info.status and index_info.status["ready"]:
                break
            time.sleep(1)

    logger.info(
        f"Index '[bold green]{index_name}[/bold green]' created successfully.",
        extra={"markup": True},
    )


def setup_database():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # These names must match what pinecone_service.py / query.py read.
    index_names = [
        os.getenv("PINECONE_SHOWS_INDEX", "shows"),
        os.getenv("PINECONE_MOVIES_INDEX", "movies"),
    ]

    existing = [index_info["name"] for index_info in pc.list_indexes()]

    for index_name in index_names:
        ensure_index(pc, index_name, existing)


if __name__ == "__main__":
    setup_database()
