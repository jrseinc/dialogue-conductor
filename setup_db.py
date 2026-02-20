import os
import time
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from rich.logging import RichHandler
from rich.console import Console

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("rich")
console = Console()


def setup_database():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "dialogue-detective")

    existing_index = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_index:
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
    else:
        logger.info(
            f"Index '[bold green]{index_name}[/bold green]' already exists.",
            extra={"markup": True},
        )


if __name__ == "__main__":
    setup_database()
