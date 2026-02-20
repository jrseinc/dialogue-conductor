import os
import tiktoken
import pysrt
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator, Optional, cast

from pinecone_service import CategoryType, ChunkMetadata, VectorDatabaseService

BASE_DIR = Path(__file__).resolve().parent
BM25_DATASET_DIR = (BASE_DIR / "bm25_dataset").resolve()

MEDIA_DIRS = [
    {"path": (BASE_DIR / "shows").resolve(), "category": "series"},
    {"path": (BASE_DIR / "movies").resolve(), "category": "movies"},
]

tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class EpisodeMetadata:
    title: str
    season: int
    episode: int
    episode_title: Optional[str] = None


def get_episode_metadata(filename: str) -> Optional[EpisodeMetadata]:
    # 1. Standardize the string: replace dots/underscores with spaces
    # This makes "Breaking.Bad" and "Breaking_Bad" look like "Breaking Bad"
    clean_name = re.sub(r"[._]", " ", filename)

    patterns = [  # Pattern: Title - S01E02 - Ep Name
        r"(?P<Title>.*?)\s*-\s*S(?P<Season>\d+)E(?P<Episode>\d+)\s*-\s*(?P<EpTitle>.*)",
        # Pattern: Title.S01E02.Quality
        r"(?P<Title>.*?)\s*S(?P<Season>\d+)E(?P<Episode>\d+)",
        # Pattern: Title.1x02.Quality
        r"(?P<Title>.*?)\s*(?P<Season>\d+)x(?P<Episode>\d+)",
        # Pattern: Title Season 1 Episode 2
        r"(?P<Title>.*?)\s*Season\s*(?P<Season>\d+)\s*Episode\s*(?P<Episode>\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_name, re.IGNORECASE)
        if match:
            data = match.groupdict()

            ep_title = (
                data.get("EpTitle", "").split("1080p")[
                    0].split("Bluray")[0].strip()
            )

            return EpisodeMetadata(
                title=data["Title"].strip(),
                season=int(data["Season"]),
                episode=int(data["Episode"]),
                episode_title=ep_title if ep_title else None,
            )
    return None


def chunker(subtitles: pysrt.SubRipFile, max_token: int = 500, overlap_line: int = 3):
    current_chunk = []
    current_token_count = 0

    for item in range(len(subtitles)):
        line_text = subtitles[item].text_without_tags.replace("\n", " ")
        line_tokens = len(tokenizer.encode(line_text))

        current_chunk.append(subtitles[item])
        current_token_count += line_tokens

        if current_token_count > max_token:
            yield current_chunk

            current_chunk = current_chunk[-overlap_line:]
            current_token_count = sum(
                len(tokenizer.encode(s.text_without_tags)) for s in current_chunk
            )

    if current_chunk:
        yield current_chunk


def process_media_folder(media_dir: Path, category: str) -> Iterator[dict]:
    print(f"Processing {category}: {media_dir.name} ---")

    for subtitle_file in media_dir.glob("*.srt"):
        episode_metadata = get_episode_metadata(subtitle_file.name)

        display_title = (
            episode_metadata.title if episode_metadata else subtitle_file.stem
        )
        try:
            subtitles = pysrt.open(str(subtitle_file), encoding="utf-8")
        except UnicodeDecodeError:
            subtitles = pysrt.open(str(subtitle_file), encoding="latin-1")

        for chunk_index, chunk_lines in enumerate(chunker(subtitles)):
            full_text = " ".join(
                [item.text_without_tags for item in chunk_lines])

            metadata: ChunkMetadata = {
                "category": cast(CategoryType, category),
                "source_id": media_dir.name,
                "title": display_title,
                "start": str(chunk_lines[0].start),
                "end": str(chunk_lines[-1].end),
            }

            if category == "series" and episode_metadata:
                metadata["season"] = episode_metadata.season
                metadata["episode"] = episode_metadata.episode

            yield {
                "id": f"{media_dir.name}_{display_title}_C{chunk_index}",
                "text": full_text,
                "metadata": metadata,
            }


def main():
    vector_db = VectorDatabaseService(BM25_DATASET_DIR)

    for media_dir in MEDIA_DIRS:
        root_path = media_dir["path"]
        category = media_dir["category"]

        if not root_path.exists():
            print(
                f"Skipping {category}: Directory {root_path} does not exists")
            continue

        for folder in root_path.iterdir():
            if folder.is_dir():
                media_iterator = process_media_folder(folder, category)

                vector_db.process_and_upload(
                    folder.name, media_iterator, category)


if __name__ == "__main__":
    main()
