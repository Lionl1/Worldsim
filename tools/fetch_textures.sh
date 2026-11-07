#!/usr/bin/env bash
# Downloads photo maps of Earth and the Moon for the Worlds tab into data/textures (optional, about 0.7 MB).
#   earth.jpg  NASA Blue Marble, public domain, through Wikimedia Commons
#   moon.jpg   NASA Lunar Reconnaissance Orbiter color map, public domain, NASA Scientific Visualization Studio
# Other bodies keep the procedural surface: no map with a stable address was found for them.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/textures
failed=""
sha() { (command -v sha256sum >/dev/null && sha256sum "$1" || shasum -a 256 "$1") | cut -d' ' -f1; }
fetch() { # name url sha256 page
  local dest="data/textures/$1" tmp="data/textures/$1.tmp"
  if [ -f "$dest" ] && [ "$(sha "$dest")" = "$3" ]; then echo "$1: already here"; return; fi
  echo "downloading $1"
  if ! curl -fsSL -A 'worldsim-fetch' -o "$tmp" "$2"; then
    rm -f "$tmp"
    failed="$failed\n  $1: the server refused or did not answer. By hand: $4 → save as $dest"
    return
  fi
  if [ "$(sha "$tmp")" != "$3" ]; then
    rm -f "$tmp"
    failed="$failed\n  $1: the file differs from the expected one, so it was not used. By hand: $4 → save as $dest"
    return
  fi
  mv "$tmp" "$dest"
}
fetch earth.jpg https://upload.wikimedia.org/wikipedia/commons/c/cd/Land_ocean_ice_2048.jpg d4dc80a6ef571939d0abe04a9bed3d3d1e6cd63e59514be1c5e43a6b069e6f1e https://commons.wikimedia.org/wiki/File:Land_ocean_ice_2048.jpg
fetch moon.jpg https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_2k.jpg f7130a1822681fa7512d7dcfd40db8c10b9ba4f06777910348698260ed7a2170 https://svs.gsfc.nasa.gov/4720
if [ -n "$failed" ]; then
  printf 'Not downloaded:%b\nThese bodies keep the procedural surface. worldsim works without them.\n' "$failed" >&2
  exit 1
fi
echo done
