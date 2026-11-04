#!/usr/bin/env python3
"""Builds compact data for the visualizer from open catalogs.

  python3 tools/build_data.py <path to hygdata_v41.csv> <path to open_exoplanet_catalogue.txt>

Sources:
  * HYG Database v4.1 (astronexus/HYG-Database) — CC BY-SA 4.0  → data/stars_hyg.json
  * Open Exoplanet Catalogue (oec_tables, H. Rein) — MIT          → data/exoplanets_oec.json
Download: git clone https://github.com/astronexus/HYG-Database ; git clone https://github.com/OpenExoplanetCatalogue/oec_tables
"""
import csv, json, math, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bv_to_teff(bv):
    # Ballesteros 2012
    return 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))


def stars(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                dist = float(r["dist"])
                mag = float(r["mag"])
            except ValueError:
                continue
            if dist <= 0 or dist >= 100000:
                continue
            if not (mag < 6.5 or dist < 25.0):
                continue
            ci = r["ci"]
            t = bv_to_teff(float(ci)) if ci not in ("", None) else 5800.0
            name = r["proper"] or (r["bayer"] + " " + r["con"] if r["bayer"] else "") or (("HIP " + r["hip"]) if r["hip"] else "")
            out.append([round(float(r["x"]), 3), round(float(r["y"]), 3), round(float(r["z"]), 3),
                        round(float(r["absmag"]), 2), round(mag, 2), int(min(max(t, 2000), 40000)), name.strip(), r["spect"][:6]])
    return out


def hms(s, hours):
    p = [float(x) for x in s.replace("+", "").split()]
    sign = -1 if s.strip().startswith("-") else 1
    v = abs(p[0]) + (p[1] if len(p) > 1 else 0) / 60 + (p[2] if len(p) > 2 else 0) / 3600
    return math.radians(sign * v * (15.0 if hours else 1.0))


def planets(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if "Confirmed planets" not in r["list"] or r["binaryflag"] == "3":
                continue
            f_ = lambda k: float(r[k]) if r[k] not in ("", None) else None
            mj, rj, a, p = f_("mass"), f_("radius"), f_("semimajoraxis"), f_("period")
            ms, ts, d = f_("hoststar_mass"), f_("hoststar_temperature"), f_("system_distance")
            if a is None and p is not None and ms:
                a = (ms * (p / 365.25) ** 2) ** (1 / 3)  # 3rd Kepler law
            xyz = None
            if d and r["system_rightascension"] and r["system_declination"]:
                ra, de = hms(r["system_rightascension"], True), hms(r["system_declination"], False)
                xyz = [round(d * math.cos(de) * math.cos(ra), 2), round(d * math.cos(de) * math.sin(ra), 2), round(d * math.sin(de), 2)]
            out.append({"n": r["name"], "m": mj, "r": rj, "a": a, "e": f_("eccentricity"), "p": p, "t": f_("temperature"),
                        "ms": ms, "rs": f_("hoststar_radius"), "ts": ts, "d": d, "xyz": xyz,
                        "y": int(r["discoveryyear"]) if r["discoveryyear"] else None, "meth": r["discoverymethod"]})
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    s = stars(sys.argv[1])
    json.dump({"source": "HYG Database v4.1, CC BY-SA 4.0, https://github.com/astronexus/HYG-Database",
               "fields": ["x_pc", "y_pc", "z_pc", "absmag", "mag", "teff", "name", "spect"], "stars": s},
              open(os.path.join(ROOT, "data", "stars_hyg.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    p = planets(sys.argv[2])
    json.dump({"source": "Open Exoplanet Catalogue (oec_tables), MIT, https://github.com/OpenExoplanetCatalogue/oec_tables",
               "units": {"m": "M_J", "r": "R_J", "a": "AU", "p": "days", "t": "K", "ms": "M_sun", "rs": "R_sun", "ts": "K", "d": "pc", "xyz": "pc (equatorial)"},
               "planets": p}, open(os.path.join(ROOT, "data", "exoplanets_oec.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"stars: {len(s)}, exoplanets: {len(p)}")
