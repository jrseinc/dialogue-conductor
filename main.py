import tiktoken
import pysrt
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
SUBTITLES_DIR = (BASE_DIR / "tbbt").resolve()
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
                data.get("EpTitle", "").split("1080p")[0].split("Bluray")[0].strip()
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


def main():
    for subtitle_file in SUBTITLES_DIR.glob("*.srt"):
        metadata = get_episode_metadata(subtitle_file.name)

        if not metadata:
            print(f"Skipping {subtitle_file.name}: Could not parse metadata")
            continue

        subtitles = pysrt.open(str(subtitle_file), encoding="utf-8")

        for chunk_index, chunk_lines in enumerate(chunker(subtitles)):
            full_text = " ".join([item.text_without_tags for item in chunk_lines])

            start_time = chunk_lines[0].start
            end_time = chunk_lines[-1].end

            payload = {
                "id": f"{metadata.title}_S{metadata.season}E{metadata.episode}_C{chunk_index}",
                "text": full_text,
                "metadata": {
                    "title": metadata.title,
                    "season": metadata.season,
                    "episode": metadata.episode,
                    "start": str(start_time),
                    "end": str(end_time),
                },
            }

            print(payload)


if __name__ == "__main__":
    main()
