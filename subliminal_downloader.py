#!/usr/bin/env python3
"""Download subtitles via subliminal -- the same engine Bazarr uses.

subliminal queries many providers at once (OpenSubtitles.com, Addic7ed via
Gestdown, TVSubtitles, Podnapisi, BSPlayer, ...) and picks the best match,
so you are not tied to a single site or a single daily quota.

Two modes:

  1. FILES  -- point it at your media (best matching, uses file hash):
         python subliminal_downloader.py files /path/to/Media --lang eng

  2. NAME   -- you have no video files, fetch by show/movie name:
         python subliminal_downloader.py show "The Big Bang Theory" \
             --season 1 --episodes 1-12 --out shows/the-big-bang-theory
         python subliminal_downloader.py movie "Inception" --year 2010 --out movies

Credentials (optional, improves results / raises limits) go in a .env file:
    OPENSUBTITLES_USERNAME=...
    OPENSUBTITLES_PASSWORD=...
    OPENSUBTITLES_APIKEY=...        # from opensubtitles.com -> API Consumers
    ADDIC7ED_USERNAME=...
    ADDIC7ED_PASSWORD=...

Note on OpenSubtitles.com quota: free accounts are capped at 20 downloads/day
(anonymous 5, VIP 1000). subliminal spreads requests across providers, but a
big bulk run can still exhaust the OpenSubtitles quota -- the other providers
keep working when it does.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from babelfish import Language
from dotenv import load_dotenv

from subliminal import (
    download_best_subtitles,
    region,
    save_subtitles,
    scan_videos,
)
from subliminal.video import Episode, Movie

# Providers that work without credentials, ordered roughly by usefulness.
DEFAULT_PROVIDERS = [
    "opensubtitlescom",
    "gestdown",      # Addic7ed proxy, no login
    "tvsubtitles",
    "podnapisi",
    "bsplayer",
]


def setup_cache() -> None:
    """subliminal requires a configured dogpile cache region before use."""
    region.configure(
        "dogpile.cache.dbm",
        expiration_time=3600 * 24 * 7,
        arguments={"filename": str(Path.cwd() / ".subliminal.cache.dbm")},
        replace_existing_backend=True,
    )


def provider_configs() -> dict:
    """Build per-provider credential configs from environment variables."""
    cfg: dict[str, dict] = {}
    os_user = os.getenv("OPENSUBTITLES_USERNAME")
    os_pass = os.getenv("OPENSUBTITLES_PASSWORD")
    os_key = os.getenv("OPENSUBTITLES_APIKEY")
    if os_user and os_pass:
        oc = {"username": os_user, "password": os_pass}
        if os_key:
            oc["apikey"] = os_key
        cfg["opensubtitlescom"] = oc
    a_user = os.getenv("ADDIC7ED_USERNAME")
    a_pass = os.getenv("ADDIC7ED_PASSWORD")
    if a_user and a_pass:
        cfg["addic7ed"] = {"username": a_user, "password": a_pass}
    return cfg


def parse_episode_spec(spec: str) -> list[int]:
    """Turn '1-12', '1,3,5' or '1-3,7,9-10' into a sorted list of ints."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def dotted(name: str) -> str:
    return ".".join(name.split())


def build_episode_videos(series: str, season: int, episodes: list[int], year: int | None) -> list[Episode]:
    vids = []
    for ep in episodes:
        # Trailing .mkv is a placeholder container: without a real extension,
        # save_subtitles' splitext() would mistake ".S01E01" for the extension.
        name = f"{dotted(series)}.S{season:02d}E{ep:02d}.mkv"
        vids.append(Episode(name=name, series=series, season=season, episodes=ep, year=year))
    return vids


def build_movie_video(title: str, year: int | None) -> Movie:
    base = f"{dotted(title)}.{year}" if year else dotted(title)
    return Movie(name=f"{base}.mkv", title=title, year=year)


def run(videos, languages, out_dir, providers, configs, min_score, single) -> int:
    if not videos:
        print("No videos to process.", file=sys.stderr)
        return 1

    print(f"Searching {len(providers)} providers for {len(videos)} item(s)...")
    results = download_best_subtitles(
        videos,
        languages,
        min_score=min_score,
        only_one=True,
        providers=providers,
        provider_configs=configs,
    )

    saved = missed = 0
    for video in videos:
        label = Path(video.name).stem
        subs = results.get(video, [])
        if not subs:
            print(f"  MISS  {label}")
            missed += 1
            continue
        save_subtitles(
            video,
            subs,
            single=single,
            directory=str(out_dir) if out_dir else None,
            encoding="utf-8",
        )
        prov = ", ".join(s.provider_name for s in subs)
        print(f"  OK    {label}  [{prov}]")
        saved += 1

    print(f"\nDone. saved={saved} missed={missed}")
    return 0 if saved else 2


def main() -> int:
    load_dotenv()
    setup_cache()

    # Common options usable before OR after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lang", default="eng", help="Language alpha-3 code (default: eng)")
    common.add_argument("--providers", default=",".join(DEFAULT_PROVIDERS),
                        help="Comma-separated provider list")
    common.add_argument("--min-score", type=int, default=0,
                        help="Minimum match score (0 = accept best available; raise for stricter matching)")
    common.add_argument("--out", default=None, help="Output directory (default: alongside files / current dir)")
    common.add_argument("--overwrite", action="store_true",
                        help="Re-download even if the .srt already exists (default: skip existing)")

    ap = argparse.ArgumentParser(
        description="Download subtitles via subliminal (multi-provider).", parents=[common]
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    pf = sub.add_parser("files", parents=[common], help="Scan a directory of video files (best matching).")
    pf.add_argument("path", help="File or directory to scan")

    ps = sub.add_parser("show", parents=[common], help="Fetch by series name (no video files needed).")
    ps.add_argument("series", help='Series name, e.g. "The Big Bang Theory"')
    ps.add_argument("--season", type=int, required=True)
    ps.add_argument("--episodes", required=True, help="e.g. 1-12 or 1,3,5 or 1-3,7")
    ps.add_argument("--year", type=int, default=None)

    pm = sub.add_parser("movie", parents=[common], help="Fetch by movie name (no video file needed).")
    pm.add_argument("title", help='Movie title, e.g. "Inception"')
    pm.add_argument("--year", type=int, default=None)

    args = ap.parse_args()

    try:
        languages = {Language(args.lang)}
    except Exception as e:  # noqa: BLE001
        print(f"Invalid language code '{args.lang}': {e}", file=sys.stderr)
        return 1

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    configs = provider_configs()
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    # When saving to a directory we use single=True so filenames have no .en suffix.
    single = out_dir is not None

    if args.mode == "files":
        videos = scan_videos(args.path)
        # scan_videos with a single file returns one video; normalize to list.
        if isinstance(videos, (Episode, Movie)):
            videos = [videos]
        single = False  # save alongside the video files, keep language suffix
    elif args.mode == "show":
        episodes = parse_episode_spec(args.episodes)
        videos = build_episode_videos(args.series, args.season, episodes, args.year)
    elif args.mode == "movie":
        videos = [build_movie_video(args.title, args.year)]
    else:  # pragma: no cover
        ap.error("unknown mode")

    # Resume support: skip items whose .srt already exists (name/movie modes).
    if out_dir and not args.overwrite:
        kept, skipped = [], 0
        for v in videos:
            if (out_dir / f"{Path(v.name).stem}.srt").exists():
                skipped += 1
            else:
                kept.append(v)
        if skipped:
            print(f"Skipping {skipped} already-downloaded item(s).")
        videos = kept
        if not videos:
            print("Nothing to do -- all requested subtitles already exist.")
            return 0

    return run(videos, languages, out_dir, providers, configs, args.min_score, single)


if __name__ == "__main__":
    raise SystemExit(main())