import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from pathlib import Path
from typing import Any, Optional, Literal
from pinecone_service import SimpleTokenizer

load_dotenv()
load_dotenv(".env.local", override=True)

BASE_DIR = Path(__file__).resolve().parent
BM25_DATASET_DIR = BASE_DIR / "bm25_dataset"

CategoryType = Literal["series", "movies"]


class DialogueSearcher:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        self._indexes = {
            "series": pc.Index(os.getenv("PINECONE_SHOWS_INDEX", "shows")),
            "movies": pc.Index(os.getenv("PINECONE_MOVIES_INDEX", "movies")),
        }

        global_model_path = BM25_DATASET_DIR / "global" / "bm25_model.json"
        encoder = BM25Encoder().load(str(global_model_path))
        encoder._tokenizer = SimpleTokenizer()
        self.bm25_encoder = encoder

    def get_query_vectors(self, query: str):
        dense_res = self.openai_client.embeddings.create(
            input=[query], model="text-embedding-3-small"
        )
        dense_vector = dense_res.data[0].embedding
        sparse_vector = self.bm25_encoder.encode_queries(query)
        return dense_vector, sparse_vector

    def search(
        self,
        query: str,
        category: CategoryType,
        source_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Any:
        """
        Search for dialogue across shows or movies.

        Args:
            query:     the search text
            category:  "series" or "movies"
            source_id: folder name of the specific show/movie (e.g. "the_big_bang_theory",
                       "interstellar"). Omit to search globally within the category.
            top_k:     number of results to return
        """
        index = self._indexes[category]
        dense, sparse = self.get_query_vectors(query)

        filter_dict: dict[str, Any] = {}
        if source_id:
            filter_dict["source_id"] = source_id

        results = index.query(
            vector=dense,
            sparse_vector=sparse,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None,
        )
        return results


if __name__ == "__main__":
    searcher = DialogueSearcher()

    print("=== Show search (TBBT, scoped) ===")
    results = searcher.search(
        query="relax no one is gonna look at her hair",
        category="series",
        source_id="the_big_bang_theory",
    )
    for match in results["matches"]:
        meta = match["metadata"]
        print(f"  [{match['score']:.3f}] S{meta.get('season', '?')}E{meta.get('episode', '?')} "
              f"{meta.get('start')} → {meta.get('end')}")
        print(f"  {meta.get('text', '')[:80]}")

    print("\n=== Movie search (global) ===")
    results = searcher.search(
        query="love transcends time and space",
        category="movies",
    )
    for match in results["matches"]:
        meta = match["metadata"]
        print(f"  [{match['score']:.3f}] {meta.get('title')} {meta.get('start')} → {meta.get('end')}")
        print(f"  {meta.get('text', '')[:80]}")

    print("\n=== Movie search (scoped to Interstellar) ===")
    results = searcher.search(
        query="wormhole five dimensions",
        category="movies",
        source_id="interstellar",
    )
    for match in results["matches"]:
        meta = match["metadata"]
        print(f"  [{match['score']:.3f}] {meta.get('title')} {meta.get('start')} → {meta.get('end')}")
        print(f"  {meta.get('text', '')[:80]}")
