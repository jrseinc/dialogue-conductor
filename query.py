import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from pathlib import Path
from typing import Any, Optional

load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parent
BM25_DATASET_DIR = BASE_DIR / "bm25_dataset"


class DialogueSearcher:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(
            os.getenv("PINECONE_INDEX_NAME", "dialogue-detective"))

    def get_query_vectors(self, query: str, source_id: str):
        # 1. Generate Dense Vector (Semantic)
        dense_res = self.openai_client.embeddings.create(
            input=[query], model="text-embedding-3-small"
        )
        dense_vector = dense_res.data[0].embedding

        # 2. Generate Sparse Vector (Keyword)
        # We load the BM25 model for the specific show to get accurate keywords
        bm25_path = BM25_DATASET_DIR / source_id / "bm25_model.json"
        bm25_encoder = BM25Encoder().load(str(bm25_path))
        sparse_vector = bm25_encoder.encode_queries(query)

        return dense_vector, sparse_vector

    def search(
        self, query: str, source_id: str, top_k: int = 5, category: Optional[str] = None
    ):
        dense, sparse = self.get_query_vectors(query, source_id)

        # Build Metadata Filter
        filter_dict: dict[str, Any] = {"source_id": source_id}
        if category:
            filter_dict["category"] = category

        # Execute Query
        results = self.index.query(
            namespace="global",  # Querying the global data lake
            vector=dense,
            sparse_vector=sparse,
            top_k=top_k,
            include_metadata=True,
        )
        return results


# Example usage
if __name__ == "__main__":
    searcher = DialogueSearcher()
    # Search for a famous Sheldon line
    results = searcher.search(
        query="relax noone is gonna look at her hair",
        source_id="the_big_bang_theory",
        category="series",
    )

    for match in results["matches"]:
        meta = match["metadata"]
        print(
            f"[{match['score']:.2f}] {meta['title']} (s{meta['season']}e{meta['episode']}):"
        )
        print(
            f"Text: {meta['text']}\nTime: {meta['start']} --> {meta['end']}\n")
