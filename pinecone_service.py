import os
from typing import Iterator, TypedDict, cast, Any, Literal
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from rich.logging import RichHandler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("rich")

CategoryType = Literal["series", "movies", "podcasts", "misc"]


class ChunkMetadata(TypedDict, total=False):
    category: CategoryType
    source_id: str
    series_folder: str
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

        index_name = os.getenv("PINECONE_INDEX_NAME", "dialogue-detective")
        self.pinecone_index = pc.Index(index_name)

        self.bm25_dataset_dir = bm25_dataset_dir
        self.batch_size = 100

    def _upload_batch(
        self, batch: list[BatchItem], bm25_encoder: BM25Encoder, namespace: str
    ) -> None:
        if not batch:
            return

        logger.info(
            f"Uploading batch of {len(batch)} chunks to namespace '{namespace}'..."
        )

        texts: list[str] = [str(item["text"]) for item in batch]

        res = self.openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
        )

        dense_vectors: list[list[float]] = [
            record.embedding for record in res.data]

        sparse_vectors: list[SparseVector] = cast(
            list[SparseVector], bm25_encoder.encode_documents(texts)
        )

        upserts: list[PineconeUpsert] = []
        for i, item in enumerate(batch):
            updated_metadata: ChunkMetadata = {
                **item["metadata"],
                "text": item["text"],
            }

            upserts.append(
                {
                    "id": item["id"],
                    "values": dense_vectors[i],
                    "sparse_values": sparse_vectors[i],
                    "metadata": updated_metadata,
                }
            )

        print("FAAK")
        self.pinecone_index.upsert(vectors=cast(
            list, upserts), namespace=namespace)

    def process_and_upload(
        self,
        source_id: str,
        payload_generator: Iterator[dict[str, Any]],
        category: CategoryType,
    ):

        bm25_path = self.bm25_dataset_dir / source_id / "bm25_model.json"

        if not bm25_path.exists():
            logger.error(
                f"BM25 model not found for {source_id} at {bm25_path}. Run training script first"
            )
            return

        logger.info(
            f"Loading BM25 model for [bold cyan]{source_id}[/bold cyan]",
            extra={"markup": True},
        )
        bm25_encoder = BM25Encoder().load(str(bm25_path))

        batch = []

        for payload in payload_generator:
            batch.append(payload)
            if len(batch) >= self.batch_size:
                self._upload_batch(batch, bm25_encoder, namespace="global")
                batch = []

        if batch:
            self._upload_batch(batch, bm25_encoder, namespace="global")
