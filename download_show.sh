#!/usr/bin/env bash
# Download English subtitles for a whole show by NAME, via subliminal_downloader.py.
#
# Usage:   ./download_show.sh "Series Name" <eps_S1> <eps_S2> ...
# Example: ./download_show.sh "The Big Bang Theory" 17 23 23 24 24 24 24 24 24 24 24 24
#          ./download_show.sh "Friends" 24 24 25 24 24 25 24 24 24 24 18
#
# Each season's count is one positional arg, in order. Files land in
# shows/<slug>/S01, S02, ...  Re-running skips episodes already downloaded.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 \"Series Name\" <eps_S1> <eps_S2> ..." >&2
  exit 1
fi

SERIES="$1"; shift
# slug: lowercase, spaces -> dashes, strip anything else
SLUG=$(echo "$SERIES" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9-')
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

season=0
for n in "$@"; do
  season=$((season + 1))
  printf '\n==================== %s S%02d (1-%s) ====================\n' "$SERIES" "$season" "$n"
  "$PY" subliminal_downloader.py show "$SERIES" \
      --season "$season" --episodes "1-$n" \
      --out "shows/$SLUG/$(printf 'S%02d' "$season")"
done

echo "DONE: $SERIES -> shows/$SLUG/"