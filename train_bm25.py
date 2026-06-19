from main import chunker
from pathlib import Path
import logging
import pysrt
from pinecone_text.sparse import BM25Encoder
from pinecone_service import SimpleTokenizer
from rich.logging import RichHandler

BASE_DIR = Path(__file__).resolve().parent
BM25_DATASET_DIR = BASE_DIR / "bm25_dataset"
GLOBAL_MODEL_DIR = BM25_DATASET_DIR / "global"

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

    srt_files = sorted(media_dir.rglob("*.srt"))
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


def build_global_corpus() -> list[str]:
    corpus: list[str] = []

    for media_dir in MEDIA_DIRS:
        root_path = media_dir["path"]
        category = media_dir["category"]

        if not root_path.exists():
            logger.error(f"Path for {category} not found: {root_path}")
            continue

        for item in root_path.iterdir():
            if item.is_dir():
                logger.info(
                    f"Adding [bold cyan]{item.name}[/bold cyan] to global corpus",
                    extra={"markup": True},
                )
                corpus.extend(create_show_corpus(item))

    return corpus


def train_and_save_global(corpus: list[str]):
    if not corpus:
        logger.warning("Global corpus is empty. Nothing to train.")
        return

    GLOBAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = GLOBAL_MODEL_DIR / "bm25_model.json"

    logger.info(
        f"Training global BM25 on [bold yellow]{len(corpus)}[/bold yellow] chunks "
        "across all shows and movies...",
        extra={"markup": True},
    )

    # stem/stopwords are disabled so the query-side tokenizer reduces to
    # lowercase + punctuation removal + whitespace split, which can be
    # reproduced exactly in TypeScript on the detective side. This keeps the
    # hosted query encoder and the indexed documents byte-for-byte consistent
    # across both Hindi (Devanagari) and English content.
    encoder = BM25Encoder(stem=False, remove_stopwords=False)
    encoder._tokenizer = SimpleTokenizer()
    encoder.fit(corpus)

    encoder.dump(str(model_path))
    logger.info(
        f"Successfully saved global model to [bold green]{model_path}[/bold green]",
        extra={"markup": True},
    )


def build_bm25_datasets():
    corpus = build_global_corpus()
    train_and_save_global(corpus)


if __name__ == "__main__":
    build_bm25_datasets()
