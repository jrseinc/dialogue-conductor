import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, cast

import pysrt
import tiktoken

from pinecone_service import CategoryType, ChunkMetadata, VectorDatabaseService

BASE_DIR = Path(__file__).resolve().parent
BM25_DATASET_DIR = (BASE_DIR / "bm25_dataset").resolve()

MEDIA_DIRS = [
    {"path": (BASE_DIR / "shows").resolve(), "category": "series"},
    {"path": (BASE_DIR / "movies").resolve(), "category": "movies"},
]

tokenizer = tiktoken.get_encoding("cl100k_base")


def slugify(name: str) -> str:
    """Normalize a media folder name to a lowercase kebab-case source_id.

    Enforces one convention across shows and movies regardless of how the
    folder happens to be named: "3_idiots", "The Big Bang Theory" both become
    "3-idiots" / "the-big-bang-theory". This is the id the detective filters
    on, so it must be stable and consistent.
    """
    slug = name.strip().lower()
    slug = re.sub(r"[\s_]+", "-", slug)       # spaces / underscores -> hyphen
    slug = re.sub(r"[^a-z0-9-]+", "", slug)   # drop anything else
    slug = re.sub(r"-{2,}", "-", slug).strip("-")  # collapse + trim hyphens
    return slug


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
                data.get("EpTitle", "").split("1080p")[0].split("Bluray")[0].strip()
            )

            return EpisodeMetadata(
                title=data["Title"].strip(),
                season=int(data["Season"]),
                episode=int(data["Episode"]),
                episode_title=ep_title if ep_title else None,
            )
    return None


def chunker(
    subtitles: pysrt.SubRipFile, lines_per_chunk: int = 3, overlap_line: int = 1
):
    subtitles_list = list(subtitles)

    step = max(1, lines_per_chunk - overlap_line)

    for i in range(0, len(subtitles_list), step):
        chunk = subtitles_list[i : i + lines_per_chunk]

        if chunk:
            yield chunk

            if i + lines_per_chunk >= len(subtitles_list):
                break


def process_media_folder(media_dir: Path, category: str) -> Iterator[dict]:
    source_id = slugify(media_dir.name)
    print(f"Processing {category}: {media_dir.name} (source_id={source_id}) ---")

    for subtitle_file in sorted(media_dir.rglob("*.srt")):
        episode_metadata = get_episode_metadata(subtitle_file.name)

        folder_title = media_dir.name.replace("_", " ").replace("-", " ").title()
        display_title = episode_metadata.title if episode_metadata else folder_title
        try:
            subtitles = pysrt.open(str(subtitle_file), encoding="utf-8")
        except UnicodeDecodeError:
            subtitles = pysrt.open(str(subtitle_file), encoding="latin-1")

        for chunk_index, chunk_lines in enumerate(chunker(subtitles)):
            full_text = " ".join([item.text_without_tags for item in chunk_lines])

            metadata: ChunkMetadata = {
                "category": cast(CategoryType, category),
                "source_id": source_id,
                "title": display_title,
                "start": str(chunk_lines[0].start),
                "end": str(chunk_lines[-1].end),
            }

            if category == "series" and episode_metadata:
                unique_id = f"{source_id}_S{episode_metadata.season:02d}E{episode_metadata.episode:02d}_C{chunk_index}"
                metadata["season"] = episode_metadata.season
                metadata["episode"] = episode_metadata.episode
            else:
                unique_id = f"{source_id}_{subtitle_file.stem}_C{chunk_index}"

            yield {
                "id": unique_id,
                "text": full_text,
                "metadata": metadata,
            }


def main():
    vector_db = VectorDatabaseService(BM25_DATASET_DIR)

    for media_dir in MEDIA_DIRS:
        root_path = media_dir["path"]
        category = media_dir["category"]

        if not root_path.exists():
            print(f"Skipping {category}: Directory {root_path} does not exists")
            continue

        for folder in root_path.iterdir():
            if folder.is_dir():
                media_iterator = process_media_folder(folder, category)

                vector_db.process_and_upload(
                    slugify(folder.name), media_iterator, category
                )


if __name__ == "__main__":
    main()
