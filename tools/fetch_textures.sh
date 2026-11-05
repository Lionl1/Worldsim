#!/usr/bin/env bash
# Downloads photo maps of Earth and the Moon for the Worlds tab into data/textures (optional, about 0.7 MB).
#   earth.jpg  NASA Blue Marble, public domain, through Wikimedia Commons
#   moon.jpg   NASA Lunar Reconnaissance Orbiter color map, public domain, NASA Scientific Visualization Studio
# Other bodies keep the procedural surface: no map with a stable address was found for them.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/textures
fetch() { # name url sha256
  echo "downloading $1"
  curl -fsSL -A 'worldsim-fetch' -o "data/textures/$1.tmp" "$2"
  got=$( (command -v sha256sum >/dev/null && sha256sum "data/textures/$1.tmp" || shasum -a 256 "data/textures/$1.tmp") | cut -d' ' -f1)
  if [ "$got" != "$3" ]; then rm -f "data/textures/$1.tmp"; echo "$1: unexpected content ($got)" >&2; exit 1; fi
  mv "data/textures/$1.tmp" "data/textures/$1"
}
fetch earth.jpg https://upload.wikimedia.org/wikipedia/commons/c/cd/Land_ocean_ice_2048.jpg d4dc80a6ef571939d0abe04a9bed3d3d1e6cd63e59514be1c5e43a6b069e6f1e
fetch moon.jpg https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_2k.jpg f7130a1822681fa7512d7dcfd40db8c10b9ba4f06777910348698260ed7a2170
echo done
