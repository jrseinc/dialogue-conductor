#!/usr/bin/env python3
"""Download subtitles from my-subs.co for a whole TV show.

Give it the URL of a show's "all seasons" page, e.g.

    https://my-subs.co/showlistsubtitles-2093-the-big-bang-theory

and it walks every season/episode, picks the subtitles in the chosen
language (English by default) and saves the .srt files locally.

How the site works (reverse-engineered, no API):
  1. The show page lists every episode as  /versions-<id>-<season>-<ep>-<slug>-subtitles
  2. Each episode page lists subtitle versions; the language is marked with a
     <span class="flag-icon flag-icon-XX" title="english"> and a /downloads/<token> link.
  3. /downloads/<token> is a 10s countdown "gate" page whose JS holds the real
     link:  REAL_URL="/download-<n>".  The server serves it immediately.
  4. /download-<n> returns the actual .srt (filename in Content-Disposition).

Usage:
    python subtitle_scraper.py <show_url> [options]

Examples:
    python subtitle_scraper.py https://my-subs.co/showlistsubtitles-2093-the-big-bang-theory
    python subtitle_scraper.py <url> --out shows/tbbt --season 1 --season 2
    python subtitle_scraper.py <url> --all-versions --lang english
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
from html import unescape
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# HTTP                                                                         #
# --------------------------------------------------------------------------- #
def fetch(url: str, referer: str | None = None) -> tuple[bytes, dict[str, str]]:
    """Return (body, headers) for a GET request with a browser-like UA."""
    headers = {"User-Agent": UA, "Accept-Encoding": "gzip"}
    if referer:
        headers["Referer"] = referer
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        body = resp.read()
        meta = {k.lower(): v for k, v in resp.headers.items()}
    if meta.get("content-encoding") == "gzip":
        body = gzip.decompress(body)
    return body, meta


def fetch_text(url: str, referer: str | None = None) -> str:
    body, _ = fetch(url, referer)
    return body.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
EPISODE_RE = re.compile(
    r"/versions-(\d+)-(\d+)-(\d+)-([a-z0-9-]+)-subtitles", re.IGNORECASE
)
DOWNLOAD_RE = re.compile(r"href='(/downloads/[^']+)'")
LANG_RE = re.compile(r'flag-icon-[a-z]+"\s+title="([^"]+)"', re.IGNORECASE)
DOWNLOADS_RE = re.compile(r"Downloads\s*:</b>\s*([\d,]+)")
VERSION_RE = re.compile(r"Version:</b>\s*<i>(.*?)</i>", re.IGNORECASE | re.DOTALL)
REAL_URL_RE = re.compile(r'REAL_URL\s*=\s*"([^"]+)"')


def parse_episodes(html: str, base_url: str) -> list[dict]:
    """Extract a de-duplicated, ordered list of episodes from a show page."""
    seen: dict[tuple[int, int], dict] = {}
    for m in EPISODE_RE.finditer(html):
        show_id, season, episode, slug = m.groups()
        key = (int(season), int(episode))
        if key not in seen:
            seen[key] = {
                "show_id": show_id,
                "season": int(season),
                "episode": int(episode),
                "slug": slug,
                "url": urljoin(base_url, m.group(0)),
            }
    return [seen[k] for k in sorted(seen)]


def _last_before(pattern: re.Pattern, html: str, pos: int) -> str | None:
    """Return the last regex group-1 match that ends at or before `pos`."""
    found = None
    for m in pattern.finditer(html, 0, pos):
        found = m.group(1)
    return found


def parse_subtitles(html: str, base_url: str) -> list[dict]:
    """Extract subtitle entries from an episode page.

    Within an entry the markup order is: Version -> language flag ->
    Downloads count -> /downloads/<token> link, so each download link is
    paired with the nearest preceding version/language/count.
    """
    entries = []
    for m in DOWNLOAD_RE.finditer(html):
        pos = m.start()
        lang = _last_before(LANG_RE, html, pos)
        downloads = _last_before(DOWNLOADS_RE, html, pos)
        version = _last_before(VERSION_RE, html, pos)
        entries.append(
            {
                "language": (lang or "").strip().lower(),
                "version": unescape((version or "").strip()),
                "downloads": int(downloads.replace(",", "")) if downloads else 0,
                "gate_url": urljoin(base_url, m.group(1)),
            }
        )
    return entries


def resolve_final_url(gate_url: str, base_url: str) -> str | None:
    """Follow a /downloads/<token> gate page to the real /download-<n> link."""
    html = fetch_text(gate_url, referer=base_url)
    m = REAL_URL_RE.search(html)
    if not m:
        return None
    return urljoin(base_url, m.group(1).replace("\\/", "/"))


# --------------------------------------------------------------------------- #
# Download                                                                     #
# --------------------------------------------------------------------------- #
def sanitize(name: str) -> str:
    name = re.sub(r"[^\w.\- ]+", "_", name).strip()
    return re.sub(r"\s+", " ", name)


def is_valid_srt(body: bytes) -> bool:
    """Reject empty/placeholder subs (the site sometimes serves a 1-cue watermark)."""
    return body.decode("utf-8", errors="replace").count("-->") >= 3


def build_filename(ep: dict, entry: dict) -> str:
    """Name files from OUR known season/episode -- the site's Content-Disposition
    names are unreliable (it labels every file S<ep>E01)."""
    tag = f"S{ep['season']:02d}E{ep['episode']:02d}"
    ver = sanitize(entry["version"]) or "sub"
    return f"{ep['slug']}.{tag}.{ver}.srt"


def download_one(
    entry: dict, ep: dict, out_dir: Path, referer: str, base_url: str, overwrite: bool
) -> tuple[str, bool]:
    """Resolve, fetch and save one subtitle entry.

    Returns (status_string, is_valid). is_valid is False for missing links or
    empty/placeholder subtitles so the caller can fall back to another version.
    """
    filename = build_filename(ep, entry)
    dest = out_dir / filename
    if dest.exists() and not overwrite:
        return f"skip (exists): {filename}", True

    final = resolve_final_url(entry["gate_url"], base_url)
    if not final:
        return "no download link", False

    body, _ = fetch(final, referer=referer)
    if not is_valid_srt(body):
        return f"empty/placeholder ({len(body)} bytes)", False

    out_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return f"saved: {filename} ({len(body)} bytes)", True


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def slug_from_url(url: str) -> str:
    m = re.search(r"showlistsubtitles-\d+-([a-z0-9-]+)", url, re.IGNORECASE)
    return m.group(1) if m else "show"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download subtitles for a whole show from my-subs.co"
    )
    ap.add_argument("url", help="Show 'all seasons' page URL (showlistsubtitles-...)")
    ap.add_argument(
        "-o", "--out", help="Output directory (default: shows/<slug>)", default=None
    )
    ap.add_argument(
        "--lang",
        default="english",
        help="Language to download, matched against the flag title (default: english)",
    )
    ap.add_argument(
        "--all-versions",
        action="store_true",
        help="Download every matching version per episode (default: only the most downloaded one)",
    )
    ap.add_argument(
        "--season",
        type=int,
        action="append",
        help="Only this season (repeatable). Default: all seasons.",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to pause between requests (be polite; default: 1.0)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the .srt already exists",
    )
    args = ap.parse_args()

    parsed = urlparse(args.url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    lang = args.lang.strip().lower()

    out_root = Path(args.out) if args.out else Path("shows") / slug_from_url(args.url)

    print(f"Fetching show page: {args.url}")
    try:
        show_html = fetch_text(args.url)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR fetching show page: {e}", file=sys.stderr)
        return 1

    episodes = parse_episodes(show_html, base_url)
    if args.season:
        wanted = set(args.season)
        episodes = [e for e in episodes if e["season"] in wanted]
    if not episodes:
        print("No episodes found. Is this a my-subs.co show page URL?", file=sys.stderr)
        return 1

    print(f"Found {len(episodes)} episodes. Output -> {out_root}/")
    saved = skipped = failed = 0

    for ep in episodes:
        tag = f"S{ep['season']:02d}E{ep['episode']:02d}"
        try:
            ep_html = fetch_text(ep["url"], referer=args.url)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag}  ERROR fetching episode page: {e}")
            failed += 1
            continue
        time.sleep(args.delay)

        subs = [s for s in parse_subtitles(ep_html, base_url) if s["language"] == lang]
        if not subs:
            print(f"  {tag}  no '{lang}' subtitles")
            continue

        subs.sort(key=lambda s: s["downloads"], reverse=True)
        season_dir = out_root / f"S{ep['season']:02d}"

        if args.all_versions:
            # Download every version (skip the empties).
            got = False
            for entry in subs:
                try:
                    status, ok = download_one(
                        entry, ep, season_dir, ep["url"], base_url, args.overwrite
                    )
                except Exception as e:  # noqa: BLE001
                    status, ok = f"ERROR: {e}", False
                print(f"  {tag}  [{entry['version'] or '?'}] {status}")
                if status.startswith("saved") or status.startswith("skip"):
                    saved += status.startswith("saved")
                    skipped += status.startswith("skip")
                    got = True
                time.sleep(args.delay)
            if not got:
                failed += 1
        else:
            # Take the most-downloaded version; fall back if it's empty/missing.
            success = False
            for entry in subs:
                try:
                    status, ok = download_one(
                        entry, ep, season_dir, ep["url"], base_url, args.overwrite
                    )
                except Exception as e:  # noqa: BLE001
                    status, ok = f"ERROR: {e}", False
                print(f"  {tag}  [{entry['version'] or '?'}] {status}")
                time.sleep(args.delay)
                if ok:
                    saved += status.startswith("saved")
                    skipped += status.startswith("skip")
                    success = True
                    break
            if not success:
                failed += 1

    print(f"\nDone. saved={saved} skipped={skipped} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())