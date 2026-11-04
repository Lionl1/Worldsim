//! Storing Mission Lab runs on disk: reproducibility and comparison of project versions.
//! Run folder: `spec.json`, `report.json`, `report.md`, `manifest.json`. A run cannot be overwritten by accident: the id contains the time and a fingerprint.

use crate::json::J;
use crate::mission;
use std::fs;
use std::path::{Path, PathBuf};

/// SHA-256 (FIPS 180-4) no external dependencies: mission description fingerprint in the report.
pub fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut m = data.to_vec();
    m.push(0x80);
    while m.len() % 64 != 56 { m.push(0); }
    m.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in m.chunks(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks(4).enumerate() { w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let mj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(mj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (a, b) in h.iter_mut().zip(v) { *a = a.wrapping_add(b); }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

/// id run id is safe for paths: only digits, a–f and hyphen. Anything else is an attempt to leave the folder.
fn run_dir(root: &Path, id: &str) -> Result<PathBuf, String> {
    if id.is_empty() || id.len() > 64 || !id.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c) || c == '-') { return Err(format!("bad run id: {id}")); }
    Ok(root.join(id))
}

/// Save a run and return its id.
pub fn save_run(root: &Path, spec_text: &str, report: &J) -> Result<String, String> {
    let io = |e: std::io::Error| format!("{}: {e}", root.display());
    let sha = sha256(spec_text.as_bytes());
    let created = now();
    let mut id = format!("{created}-{}", &sha[..8]);
    let mut k = 0;
    while root.join(&id).exists() { k += 1; id = format!("{created}-{}-{k}", &sha[..8]); }
    let dir = root.join(&id);
    fs::create_dir_all(&dir).map_err(io)?;
    let get = |key: &str| report.get(key).cloned().unwrap_or(J::Null);
    let manifest = J::obj().set("id", id.clone()).set("created", created).set("spec_sha256", sha.clone()).set("name", get("name")).set("task", get("task")).set("world", get("world"))
        .set("runs", get("runs")).set("seed", get("seed")).set("generator", get("generator")).set("success", get("success")).set("requirements_met", get("requirements_met")).set("requirements_total", get("requirements_total"));
    fs::write(dir.join("spec.json"), spec_text).map_err(io)?;
    fs::write(dir.join("report.json"), report.render()).map_err(io)?;
    fs::write(dir.join("report.md"), mission::markdown(report)).map_err(io)?;
    // the manifest is written last: an incomplete run does not appear in the list
    fs::write(dir.join("manifest.json"), manifest.render()).map_err(io)?;
    Ok(id)
}

pub fn list_runs(root: &Path) -> J {
    let mut out: Vec<(f64, J)> = vec![];
    if let Ok(rd) = fs::read_dir(root) {
        for e in rd.flatten() {
            if let Some(m) = fs::read_to_string(e.path().join("manifest.json")).ok().and_then(|t| J::parse(&t).ok()) {
                out.push((m.get("created").and_then(|x| x.as_f64()).unwrap_or(0.0), m));
            }
        }
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    J::Arr(out.into_iter().take(500).map(|x| x.1).collect())
}

pub fn load_report(root: &Path, id: &str) -> Result<J, String> {
    let t = fs::read_to_string(run_dir(root, id)?.join("report.json")).map_err(|_| format!("run {id} not found"))?;
    J::parse(&t)
}

pub fn load_spec(root: &Path, id: &str) -> Result<J, String> {
    let t = fs::read_to_string(run_dir(root, id)?.join("spec.json")).map_err(|_| format!("run {id} not found"))?;
    J::parse(&t)
}

fn flatten(prefix: &str, j: &J, out: &mut Vec<(String, String)>) {
    match j {
        J::Obj(v) => for (k, x) in v { flatten(&if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") }, x, out); },
        _ => out.push((prefix.to_string(), j.render())),
    }
}

/// What changed between two runs: success, requirements and mission description lines.
pub fn compare(root: &Path, a: &str, b: &str) -> Result<J, String> {
    let (ra, rb) = (load_report(root, a)?, load_report(root, b)?);
    let n = |r: &J, k: &str| r.get(k).and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
    let empty: &[J] = &[];
    let (sa, sb) = (ra.get("scorecard").and_then(|x| x.as_arr()).unwrap_or(empty), rb.get("scorecard").and_then(|x| x.as_arr()).unwrap_or(empty));
    let txt = |r: &J, k: &str| r.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let reqs: Vec<J> = sa.iter().map(|q| {
        let id = txt(q, "id");
        let other = sb.iter().find(|x| txt(x, "id") == id);
        J::obj().set("id", id).set("text", txt(q, "text")).set("a", n(q, "value")).set("a_verdict", txt(q, "verdict"))
            .set("b", other.map(|x| J::from(n(x, "value"))).unwrap_or(J::Null)).set("b_verdict", other.map(|x| txt(x, "verdict")).unwrap_or("not in run".into()))
    }).collect();
    let (mut fa, mut fb) = (vec![], vec![]);
    flatten("", &load_spec(root, a)?, &mut fa);
    flatten("", &load_spec(root, b)?, &mut fb);
    let mut keys: Vec<&String> = fa.iter().chain(&fb).map(|x| &x.0).collect();
    keys.sort();
    keys.dedup();
    let find = |f: &[(String, String)], k: &str| f.iter().find(|x| x.0 == k).map(|x| x.1.clone());
    let changes: Vec<J> = keys.into_iter().filter_map(|k| {
        let (x, y) = (find(&fa, k), find(&fb, k));
        (x != y).then(|| J::obj().set("key", k.clone()).set("a", x.unwrap_or("—".into())).set("b", y.unwrap_or("—".into())))
    }).collect();
    Ok(J::obj().set("a", a).set("b", b).set("success_a", n(&ra, "success")).set("success_b", n(&rb, "success")).set("delta", n(&rb, "success") - n(&ra, "success"))
        .set("requirements", J::Arr(reqs)).set("spec_changes", J::Arr(changes)))
}
