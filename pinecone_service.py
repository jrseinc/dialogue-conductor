import os
import re
import regex
from typing import Iterator, TypedDict, cast, Any, Literal
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from rich.logging import RichHandler

load_dotenv()
load_dotenv(".env.local", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("rich")

CategoryType = Literal["series", "movies", "podcasts", "misc"]

TOKEN_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")


class SimpleTokenizer:
    """
    Shared tokenizer used on both the conductor (doc encoding) and the detective
    (query encoding via the TS port). Keeps the two sides byte-identical.

    Replaces nltk.word_tokenize so the TS port doesn't need to replicate NLTK's
    Treebank/Punkt rules, which behave inconsistently on Hindi (Devanagari).
    """

    lower_case = True
    remove_punctuation = True
    remove_stopwords = False
    stem = False
    language = "english"

    def __call__(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())


class ChunkMetadata(TypedDict, total=False):
    category: CategoryType
    source_id: str
    title: str
    season: int
    episode: int
    start: str
    end: str
    text: str


class BatchItem(TypedDict):
    id: str
    text: str
    metadata: ChunkMetadata


class SparseVector(TypedDict):
    indices: list[int]
    values: list[float]


class PineconeUpsert(TypedDict):
    id: str
    values: list[float]
    sparse_values: SparseVector
    metadata: ChunkMetadata


class VectorDatabaseService:
    def __init__(self, bm25_dataset_dir: Path):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        shows_index_name = os.getenv("PINECONE_SHOWS_INDEX", "shows")
        movies_index_name = os.getenv("PINECONE_MOVIES_INDEX", "movies")

        self._indexes: dict[str, Any] = {
            "series": pc.Index(shows_index_name),
            "movies": pc.Index(movies_index_name),
        }

        self.bm25_dataset_dir = bm25_dataset_dir
        self.batch_size = 100

        global_model_path = bm25_dataset_dir / "global" / "bm25_model.json"
        if not global_model_path.exists():
            raise FileNotFoundError(
                f"Global BM25 model not found at {global_model_path}. "
                "Run train_bm25.py first."
            )

        logger.info(
            f"Loading global BM25 model from [bold cyan]{global_model_path}[/bold cyan]",
            extra={"markup": True},
        )
        encoder = BM25Encoder().load(str(global_model_path))
        encoder._tokenizer = SimpleTokenizer()
        self.bm25_encoder = encoder

    def _get_index(self, category: str) -> Any:
        index = self._indexes.get(category)
        if index is None:
            raise ValueError(
                f"No Pinecone index configured for category '{category}'. "
                f"Expected one of: {list(self._indexes.keys())}"
            )
        return index

    def _upload_batch(self, batch: list[BatchItem], index: Any) -> None:
        if not batch:
            return

        logger.info(f"Uploading batch of {len(batch)} chunks...")

        texts: list[str] = [str(item["text"]) for item in batch]

        res = self.openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
        )

        dense_vectors: list[list[float]] = [record.embedding for record in res.data]

        sparse_vectors: list[SparseVector] = cast(
            list[SparseVector], self.bm25_encoder.encode_documents(texts)
        )

        upserts: list[PineconeUpsert] = []
        for i, item in enumerate(batch):
            upserts.append(
                {
                    "id": item["id"],
                    "values": dense_vectors[i],
                    "sparse_values": sparse_vectors[i],
                    "metadata": {**item["metadata"], "text": item["text"]},
                }
            )

        index.upsert(vectors=cast(list, upserts))

    def process_and_upload(
        self,
        source_id: str,
        payload_generator: Iterator[dict[str, Any]],
        category: CategoryType,
    ) -> None:
        index = self._get_index(category)

        logger.info(
            f"Uploading [bold cyan]{source_id}[/bold cyan] "
            f"(category={category}) → index '{index._config.host.split('.')[0]}'",
            extra={"markup": True},
        )

        batch: list[BatchItem] = []

        for payload in payload_generator:
            batch.append(payload)
            if len(batch) >= self.batch_size:
                self._upload_batch(batch, index)
                batch = []

        if batch:
            self._upload_batch(batch, index)
