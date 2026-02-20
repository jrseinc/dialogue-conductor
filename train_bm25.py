from main import chunker
from pathlib import Path
import logging
import pysrt
from pinecone_text.sparse import BM25Encoder
from rich.logging import RichHandler

BASE_DIR = Path(__file__).resolve().parent
BM25_DATASET_DIR = BASE_DIR / "bm25_dataset"

MEDIA_DIRS = [
    {"path": BASE_DIR / "shows", "category": "series"},
    {"path": BASE_DIR / "movies", "category": "movies"},
]

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("rich")


def create_show_corpus(media_dir: Path) -> list[str]:
    corpus = []

    srt_files = list(media_dir.glob("*.srt"))
    logger.info(f"Found {len(srt_files)} subtitle files in {media_dir.name}")

    for subtitle_file in srt_files:
        try:
            try:
                subtitles = pysrt.open(str(subtitle_file), encoding="utf-8")
            except UnicodeDecodeError:
                subtitles = pysrt.open(str(subtitle_file), encoding="latin-1")

            for chunk_lines in chunker(subtitles):
                full_text = " ".join(
                    [item.text_without_tags for item in chunk_lines])
                corpus.append(full_text)
        except Exception as e:
            logger.warning(f"Error reading {subtitle_file.name}: {e}")

    return corpus


def train_and_save_bm25(media_dir: Path):
    logger.info(
        f"Building corpus for: [bold cyan]{media_dir.name}[/bold cyan]",
        extra={"markup": True},
    )
    corpus = create_show_corpus(media_dir)

    if not corpus:
        logger.warning(f"No subtitles found in {media_dir.name}. Skipping.")
        return

    destination_dir = BM25_DATASET_DIR / media_dir.name
    destination_dir.mkdir(parents=True, exist_ok=True)

    model_path = destination_dir / "bm25_model.json"

    logger.info(
        f"Training BM25 on [bold yellow]{len(corpus)}[/bold yellow] chunks...",
        extra={"markup": True},
    )
    encoder = BM25Encoder()
    encoder.fit(corpus)

    encoder.dump(str(model_path))
    logger.info(
        f"Successfully saved trained model to [bold green]{model_path}[/bold green]",
        extra={"markup": True},
    )


def build_bm25_datasets():
    for media_dir in MEDIA_DIRS:
        root_path = media_dir["path"]
        category = media_dir["category"]

        if not root_path.exists():
            logger.error(f"Path for {category} not found: {root_path}")
            continue

        for item in root_path.iterdir():
            if item.is_dir():
                train_and_save_bm25(item)


if __name__ == "__main__":
    build_bm25_datasets()
