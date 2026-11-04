//! Loading terrain of a real site: ESRI ASCII Grid (.asc), number table (.csv, .txt), array numpy (.npy).
//! Terrain is converted to a square grid: the map center coincides with the pad or start.

use crate::env::Terrain;
use std::path::Path;

pub const MAX_GRID: usize = 2049;

/// Build terrain from a square height grid.
pub fn terrain_from_grid(mut h: Vec<f64>, n: usize, size: f64, friction: f64) -> Result<Terrain, String> {
    if n < 3 { return Err("terrain: a grid of at least 3×3".into()); }
    if n > MAX_GRID { return Err(format!("terrain: grid {n}×{n} exceeds the limit {MAX_GRID}")); }
    if h.len() != n * n { return Err(format!("terrain: expected {} values, got {}", n * n, h.len())); }
    if !size.is_finite() || size <= 0.0 { return Err("terrain: map size must be greater than zero".into()); }
    if !(0.0..=2.0).contains(&friction) { return Err("terrain: friction out of range [0, 2]".into()); }
    // isolated data holes are filled from neighbors; a map with almost no data is an error, not invented terrain
    let bad = h.iter().filter(|x| !x.is_finite()).count();
    if bad * 5 > h.len() { return Err(format!("terrain: no data in {:.0}% points", 100.0 * bad as f64 / h.len() as f64)); }
    if bad > 0 {
        let good = h.iter().filter(|x| x.is_finite()).sum::<f64>() / (h.len() - bad) as f64;
        for _ in 0..n {
            let old = h.clone();
            let mut left = 0;
            for j in 0..n { for i in 0..n {
                if old[j * n + i].is_finite() { continue; }
                let mut s = (0.0, 0);
                for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (a, b) = (i as i64 + di, j as i64 + dj);
                    if a >= 0 && b >= 0 && (a as usize) < n && (b as usize) < n { let v = old[b as usize * n + a as usize]; if v.is_finite() { s.0 += v; s.1 += 1; } }
                }
                if s.1 > 0 { h[j * n + i] = s.0 / s.1 as f64; } else { left += 1; }
            }}
            if left == 0 { break; }
        }
        for x in &mut h { if !x.is_finite() { *x = good; } }
    }
    Ok(Terrain { n, size, h, friction })
}

/// Rectangular grid → central square with an odd side (the center falls on a node).
fn center_square(h: &[f64], cols: usize, rows: usize) -> (Vec<f64>, usize) {
    let mut n = cols.min(rows);
    if n % 2 == 0 { n -= 1; }
    let (ox, oy) = ((cols - n) / 2, (rows - n) / 2);
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n { for i in 0..n { out.push(h[(j + oy) * cols + i + ox]); } }
    (out, n)
}

/// ESRI ASCII Grid: header ncols, nrows, cellsize, NODATA_value, then rows from top to bottom (north is the first row).
pub fn parse_asc(text: &str, friction: f64) -> Result<Terrain, String> {
    let (mut cols, mut rows, mut cell, mut nodata) = (0usize, 0usize, 0.0f64, f64::NAN);
    let mut values: Vec<f64> = vec![];
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() { continue; }
        let first = t.split_whitespace().next().unwrap_or("");
        if first.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
            let mut it = t.split_whitespace();
            let (k, v) = (it.next().unwrap_or("").to_ascii_lowercase(), it.next().unwrap_or(""));
            match k.as_str() {
                "ncols" => cols = v.parse().map_err(|_| "asc: bad ncols")?,
                "nrows" => rows = v.parse().map_err(|_| "asc: bad nrows")?,
                "cellsize" => cell = v.parse().map_err(|_| "asc: bad cellsize")?,
                "nodata_value" => nodata = v.parse().map_err(|_| "asc: bad NODATA_value")?,
                "xllcorner" | "yllcorner" | "xllcenter" | "yllcenter" => {}
                _ => return Err(format!("asc: unknown header {k}")),
            }
        } else {
            for v in t.split_whitespace() {
                let x: f64 = v.parse().map_err(|_| format!("asc: not a number '{v}'"))?;
                values.push(if x == nodata { f64::NAN } else { x });
            }
        }
    }
    if cols == 0 || rows == 0 || cell <= 0.0 { return Err("asc: need ncols, nrows and cellsize".into()); }
    if values.len() != cols * rows { return Err(format!("asc: expected {} values, got {}", cols * rows, values.len())); }
    // in the file the first row is north, in Terrain row 0 is south
    let mut flipped = vec![0.0; values.len()];
    for r in 0..rows { flipped[(rows - 1 - r) * cols..(rows - r) * cols].copy_from_slice(&values[r * cols..(r + 1) * cols]); }
    let (h, n) = center_square(&flipped, cols, rows);
    terrain_from_grid(h, n, cell * (n - 1) as f64, friction)
}

/// Number table; the separator is a comma or space; the first row is south.
pub fn parse_csv(text: &str, size: f64, friction: f64) -> Result<Terrain, String> {
    let mut rows = 0;
    let mut cols = 0;
    let mut vals = vec![];
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: Vec<f64> = line.split(|c: char| c == ',' || c == ';' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().map_err(|_| format!("csv: not a number '{x}'"))).collect::<Result<_, _>>()?;
        if rows == 0 { cols = row.len(); } else if row.len() != cols { return Err(format!("csv: in row {} {} values, expected {cols}", rows + 1, row.len())); }
        vals.extend(row);
        rows += 1;
    }
    if rows == 0 { return Err("csv: file is empty".into()); }
    let (h, n) = center_square(&vals, cols, rows);
    terrain_from_grid(h, n, size * (n - 1) as f64 / (cols.max(rows) - 1).max(1) as f64, friction)
}

/// .npy: two-dimensional float32 or float64 array, order C.
pub fn parse_npy(b: &[u8], size: f64, friction: f64) -> Result<Terrain, String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" { return Err("npy: this is not a file numpy".into()); }
    let (hlen, start) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let head = std::str::from_utf8(b.get(start..start + hlen).ok_or("npy: truncated header")?).map_err(|_| "npy: bad header")?;
    if head.contains("'fortran_order': True") { return Err("npy: Fortran order is not supported".into()); }
    let wide = if head.contains("<f8") { true } else if head.contains("<f4") { false } else { return Err("npy: need float32 or float64".into()) };
    let shape = head.split("'shape':").nth(1).and_then(|x| x.split('(').nth(1)).and_then(|x| x.split(')').next()).ok_or("npy: no array shape")?;
    let dims: Vec<usize> = shape.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()).map(|x| x.parse().map_err(|_| "npy: bad shape")).collect::<Result<_, _>>()?;
    if dims.len() != 2 { return Err(format!("npy: need a two-dimensional array, got {}", dims.len())); }
    let (rows, cols) = (dims[0], dims[1]);
    let w = if wide { 8 } else { 4 };
    let data = b.get(start + hlen..).ok_or("npy: no data")?;
    if data.len() != rows.saturating_mul(cols).saturating_mul(w) { return Err("npy: data size does not match the shape".into()); }
    let vals: Vec<f64> = data.chunks(w).map(|c| if wide { f64::from_le_bytes(c.try_into().unwrap()) } else { f32::from_le_bytes(c.try_into().unwrap()) as f64 }).collect();
    let (h, n) = center_square(&vals, cols, rows);
    terrain_from_grid(h, n, size * (n - 1) as f64 / (cols.max(rows) - 1).max(1) as f64, friction)
}

/// Loading by extension. For .csv and .npy the map size is given explicitly: the file does not contain it.
pub fn load(path: &str, size: Option<f64>, friction: f64) -> Result<Terrain, String> {
    let ext = Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let need = || size.ok_or(format!("terrain {path}: for this format specify the map size in meters"));
    match ext.as_str() {
        "asc" => parse_asc(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?, friction),
        "csv" | "txt" => parse_csv(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?, need()?, friction),
        "npy" => parse_npy(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?, need()?, friction),
        _ => Err(format!("terrain {path}: supported formats .asc, .csv, .txt, .npy")),
    }
}
