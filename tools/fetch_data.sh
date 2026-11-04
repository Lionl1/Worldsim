#!/usr/bin/env bash
# Downloads open catalogs and builds data/stars_hyg.json and data/exoplanets_oec.json
#   HYG Database v4.1 (CC BY-SA 4.0)  https://github.com/astronexus/HYG-Database
#   Open Exoplanet Catalogue (MIT)    https://github.com/OpenExoplanetCatalogue/oec_tables
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"
echo "→ downloading Open Exoplanet Catalogue…"
git clone -q --depth 1 https://github.com/OpenExoplanetCatalogue/oec_tables.git "$TMP/oec"
echo "→ downloading HYG v4.1 (~34 MB)…"
git clone -q --depth 1 --filter=blob:none --sparse https://github.com/astronexus/HYG-Database.git "$TMP/hyg"
git -C "$TMP/hyg" sparse-checkout set hyg/CURRENT
python3 tools/build_data.py "$TMP/hyg/hyg/CURRENT/hygdata_v41.csv" "$TMP/oec/comma_separated/open_exoplanet_catalogue.txt"
rm -rf "$TMP"
echo "✓ done: data/"
