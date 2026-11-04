//! Галактики и скопления: N тел с деревом Барнса–Хата (O(N log N)), многопоточный расчёт сил.
//! Единицы: кпк, 10¹⁰ M☉, км/с; единица времени = кпк/(км/с) ≈ 0.978 млрд лет.
//! Начальные условия: сфера Пламмера (функция распределения Аарсета–Энона–Уилена),
//! галактика = живое гало Пламмера + экспоненциальный диск + балдж; столкновение двух галактик.

use crate::rng::Rng;
use crate::vec3::Vec3;
use std::f64::consts::PI;

pub const G_GAL: f64 = 43_009.1; // кпк·(км/с)²/(10¹⁰ M☉)
pub const TIME_GYR: f64 = 0.977_8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind { Halo = 0, Bulge = 1, Disk = 2, Star = 3 }

#[derive(Clone, Debug)]
pub struct Galaxy {
    pub pos: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    pub mass: Vec<f64>,
    pub kind: Vec<Kind>,
    pub eps: f64,   // смягчение, кпк
    pub theta: f64, // критерий раскрытия
    pub t: f64,
    acc: Vec<Vec3>,
}

// ---------------------------------------------------------------- октодерево
#[derive(Clone, Copy)]
struct Node {
    center: Vec3,
    half: f64,
    com: Vec3,
    mass: f64,
    child: [u32; 8], // 0 = нет
    body: i32,       // >=0 — лист с одной частицей
}

struct Tree { nodes: Vec<Node>, next: Vec<i32> }

impl Tree {
    fn build(pos: &[Vec3], mass: &[f64]) -> Tree {
        let (mut lo, mut hi) = (Vec3::new(f64::MAX, f64::MAX, f64::MAX), Vec3::new(f64::MIN, f64::MIN, f64::MIN));
        for p in pos {
            lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        let half = 0.5 * (hi.x - lo.x).max(hi.y - lo.y).max(hi.z - lo.z) * 1.0001 + 1e-9;
        let root = Node { center: (lo + hi) * 0.5, half, com: Vec3::ZERO, mass: 0.0, child: [0; 8], body: -1 };
        let mut t = Tree { nodes: Vec::with_capacity(pos.len() * 2), next: vec![-1; pos.len()] };
        t.nodes.push(root);
        for i in 0..pos.len() { t.insert(0, i, pos, 0); }
        t.summarize(0, pos, mass);
        t
    }

    fn octant(c: Vec3, p: Vec3) -> usize { ((p.x > c.x) as usize) | (((p.y > c.y) as usize) << 1) | (((p.z > c.z) as usize) << 2) }

    fn child_center(n: &Node, o: usize) -> Vec3 {
        let h = n.half * 0.5;
        n.center + Vec3::new(if o & 1 != 0 { h } else { -h }, if o & 2 != 0 { h } else { -h }, if o & 4 != 0 { h } else { -h })
    }

    // body: -1 пусто, >=0 лист (голова списка частиц), -2 внутренний узел
    fn insert(&mut self, ni: usize, i: usize, pos: &[Vec3], depth: usize) {
        match self.nodes[ni].body {
            -1 => self.nodes[ni].body = i as i32,
            b if b >= 0 => {
                if depth > 48 {
                    // совпадающие точки: держим списком в одном листе
                    self.next[i] = b;
                    self.nodes[ni].body = i as i32;
                    return;
                }
                self.nodes[ni].body = -2;
                let mut cur = b;
                while cur >= 0 {
                    let nxt = self.next[cur as usize];
                    self.next[cur as usize] = -1;
                    self.push_down(ni, cur as usize, pos, depth);
                    cur = nxt;
                }
                self.push_down(ni, i, pos, depth);
            }
            _ => self.push_down(ni, i, pos, depth),
        }
    }

    fn push_down(&mut self, ni: usize, i: usize, pos: &[Vec3], depth: usize) {
        let o = Tree::octant(self.nodes[ni].center, pos[i]);
        let c = self.nodes[ni].child[o];
        if c == 0 {
            let n = &self.nodes[ni];
            let child = Node { center: Tree::child_center(n, o), half: n.half * 0.5, com: Vec3::ZERO, mass: 0.0, child: [0; 8], body: i as i32 };
            self.nodes.push(child);
            let idx = (self.nodes.len() - 1) as u32;
            self.nodes[ni].child[o] = idx;
        } else {
            self.insert(c as usize, i, pos, depth + 1);
        }
    }

    fn summarize(&mut self, ni: usize, pos: &[Vec3], mass: &[f64]) -> (f64, Vec3) {
        let n = self.nodes[ni];
        let mut m = 0.0;
        let mut c = Vec3::ZERO;
        if n.body >= 0 {
            let mut cur = n.body;
            while cur >= 0 {
                m += mass[cur as usize];
                c += pos[cur as usize] * mass[cur as usize];
                cur = self.next[cur as usize];
            }
        } else {
            for k in 0..8 {
                let ch = n.child[k];
                if ch != 0 {
                    let (mm, cc) = self.summarize(ch as usize, pos, mass);
                    m += mm;
                    c += cc * mm;
                }
            }
        }
        let com = if m > 0.0 { c / m } else { n.center };
        self.nodes[ni].mass = m;
        self.nodes[ni].com = com;
        (m, com)
    }

    /// Ускорение и потенциал в точке p (частица `skip` исключается).
    fn field(&self, p: Vec3, skip: usize, theta2: f64, eps2: f64, posv: &[Vec3], massv: &[f64]) -> (Vec3, f64) {
        let pos_of = |j: usize| posv[j];
        let mass_of = |j: usize| massv[j];
        let mut a = Vec3::ZERO;
        let mut phi = 0.0;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if n.mass <= 0.0 { continue; }
            if n.body >= 0 {
                let mut cur = n.body;
                while cur >= 0 {
                    let j = cur as usize;
                    cur = self.next[j];
                    if j == skip { continue; }
                    let d = pos_of(j) - p;
                    let r2 = d.norm2() + eps2;
                    let inv = 1.0 / r2.sqrt();
                    a += d * (mass_of(j) * inv * inv * inv);
                    phi -= mass_of(j) * inv;
                }
                continue;
            } else {
                let d = n.com - p;
                let size = 2.0 * n.half;
                if size * size >= theta2 * d.norm2() {
                    for &c in &n.child { if c != 0 { stack.push(c); } }
                    continue;
                }
            }
            let d = n.com - p;
            let r2 = d.norm2() + eps2;
            let inv = 1.0 / r2.sqrt();
            a += d * (n.mass * inv * inv * inv);
            phi -= n.mass * inv;
        }
        (a * G_GAL, phi * G_GAL)
    }
}

// ---------------------------------------------------------------- система
impl Galaxy {
    pub fn new(pos: Vec<Vec3>, vel: Vec<Vec3>, mass: Vec<f64>, kind: Vec<Kind>, eps: f64) -> Galaxy {
        let n = pos.len();
        let mut g = Galaxy { pos, vel, mass, kind, eps, theta: 0.6, t: 0.0, acc: vec![Vec3::ZERO; n] };
        g.to_com();
        g.compute_acc();
        g
    }

    pub fn len(&self) -> usize { self.pos.len() }

    fn to_com(&mut self) {
        let m: f64 = self.mass.iter().sum();
        let mut rc = Vec3::ZERO;
        let mut vc = Vec3::ZERO;
        for i in 0..self.len() { rc += self.pos[i] * self.mass[i]; vc += self.vel[i] * self.mass[i]; }
        rc = rc / m;
        vc = vc / m;
        for i in 0..self.len() { self.pos[i] -= rc; self.vel[i] -= vc; }
    }

    fn par_field(&self, want_phi: bool) -> (Vec<Vec3>, Vec<f64>) {
        let tree = Tree::build(&self.pos, &self.mass);
        let n = self.len();
        let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4).min(16);
        let chunk = n.div_ceil(threads).max(1);
        let mut acc = vec![Vec3::ZERO; n];
        let mut phi = vec![0.0; if want_phi { n } else { 0 }];
        let (th2, e2) = (self.theta * self.theta, self.eps * self.eps);
        std::thread::scope(|s| {
            let tree = &tree;
            let pos = &self.pos;
            let mass = &self.mass;
            let mut phi_chunks: Vec<&mut [f64]> = if want_phi { phi.chunks_mut(chunk).collect() } else { Vec::new() };
            for (ci, ac) in acc.chunks_mut(chunk).enumerate() {
                let mut ph = if want_phi { Some(phi_chunks.remove(0)) } else { None };
                s.spawn(move || {
                    for (k, a) in ac.iter_mut().enumerate() {
                        let i = ci * chunk + k;
                        let (f, p) = tree.field(pos[i], i, th2, e2, pos, mass);
                        *a = f;
                        if let Some(ph) = ph.as_deref_mut() { ph[k] = p; }
                    }
                });
            }
        });
        (acc, phi)
    }

    fn compute_acc(&mut self) { self.acc = self.par_field(false).0; }

    /// Шаг leapfrog KDK (симплектический).
    pub fn step(&mut self, dt: f64) {
        for i in 0..self.len() { self.vel[i] += self.acc[i] * (0.5 * dt); }
        for i in 0..self.len() { self.pos[i] += self.vel[i] * dt; }
        self.compute_acc();
        for i in 0..self.len() { self.vel[i] += self.acc[i] * (0.5 * dt); }
        self.t += dt;
    }

    /// (кинетическая, потенциальная) энергия; потенциал — по дереву.
    pub fn energy(&self) -> (f64, f64) {
        let (_, phi) = self.par_field(true);
        let k: f64 = (0..self.len()).map(|i| 0.5 * self.mass[i] * self.vel[i].norm2()).sum();
        let w: f64 = (0..self.len()).map(|i| 0.5 * self.mass[i] * phi[i]).sum();
        (k, w)
    }

    pub fn momentum(&self) -> Vec3 { (0..self.len()).fold(Vec3::ZERO, |a, i| a + self.vel[i] * self.mass[i]) }

    /// Прямое суммирование ускорения (для проверки точности дерева).
    pub fn direct_acc(&self, i: usize) -> Vec3 {
        let e2 = self.eps * self.eps;
        let mut a = Vec3::ZERO;
        for j in 0..self.len() {
            if j == i { continue; }
            let d = self.pos[j] - self.pos[i];
            let r2 = d.norm2() + e2;
            a += d * (self.mass[j] / (r2 * r2.sqrt()));
        }
        a * G_GAL
    }

    pub fn tree_acc(&self) -> Vec<Vec3> { self.par_field(false).0 }

    /// Профиль круговой скорости √(r·|a_r|) — кривая вращения.
    /// Диапазон кривой вращения: 25 кпк для дисков, иначе от размера самой системы.
    pub fn rotation_range(&self) -> f64 {
        if self.kind.iter().any(|k| *k == Kind::Disk) { return 25.0; }
        let mut r: Vec<f64> = self.pos.iter().map(|p| (p.x * p.x + p.y * p.y).sqrt()).collect();
        r.sort_by(|a, b| a.total_cmp(b));
        3.0 * r.get(r.len() / 2).copied().unwrap_or(1.0).max(1e-9)
    }

    pub fn rotation_curve(&self, r_max: f64, bins: usize) -> Vec<(f64, f64)> {
        // без диска считаем круговую скорость по всем частицам (сфера Пламмера, шаровое скопление)
        let any_disk = self.kind.iter().any(|k| *k == Kind::Disk);
        let mut sum = vec![0.0; bins];
        let mut cnt = vec![0usize; bins];
        for i in 0..self.len() {
            if any_disk && self.kind[i] != Kind::Disk { continue; }
            let p = self.pos[i];
            let r = (p.x * p.x + p.y * p.y).sqrt();
            let b = (r / r_max * bins as f64) as usize;
            if b < bins {
                let rh = Vec3::new(p.x, p.y, 0.0) / r.max(1e-9);
                sum[b] += (-(self.acc[i].dot(rh)) * r).max(0.0).sqrt();
                cnt[b] += 1;
            }
        }
        (0..bins).filter(|&b| cnt[b] > 0).map(|b| ((b as f64 + 0.5) * r_max / bins as f64, sum[b] / cnt[b] as f64)).collect()
    }
}

// ---------------------------------------------------------------- начальные условия

/// Сфера Пламмера: масса m, масштаб a, центр, скорость. Равновесная (вириальная).
pub fn plummer(rng: &mut Rng, n: usize, m: f64, a: f64, kind: Kind) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>, Vec<Kind>) {
    let mut pos = Vec::with_capacity(n);
    let mut vel = Vec::with_capacity(n);
    let rand_dir = |rng: &mut Rng| -> Vec3 {
        let z = rng.uniform(-1.0, 1.0);
        let ph = rng.uniform(0.0, 2.0 * PI);
        let s = (1.0 - z * z).sqrt();
        Vec3::new(s * ph.cos(), s * ph.sin(), z)
    };
    for _ in 0..n {
        let r = loop {
            let x = rng.f64().max(1e-10).min(0.999);
            let r = a / (x.powf(-2.0 / 3.0) - 1.0).sqrt();
            if r < 30.0 * a { break r; }
        };
        // скорость: выборка q = v/v_esc из g(q) = q²(1−q²)^{7/2} отбором
        let q = loop {
            let q = rng.f64();
            if rng.f64() * 0.1 < q * q * (1.0 - q * q).powf(3.5) { break q; }
        };
        let vesc = (2.0 * G_GAL * m / (r * r + a * a).sqrt()).sqrt();
        pos.push(rand_dir(rng) * r);
        vel.push(rand_dir(rng) * (q * vesc));
    }
    (pos, vel, vec![m / n as f64; n], vec![kind; n])
}

pub struct DiskParams {
    pub n_disk: usize,
    pub n_bulge: usize,
    pub n_halo: usize,
    pub m_disk: f64,  // 10¹⁰ M☉
    pub r_disk: f64,  // масштаб, кпк
    pub z_disk: f64,
    pub m_bulge: f64,
    pub a_bulge: f64,
    pub m_halo: f64,
    pub a_halo: f64,
}

impl DiskParams {
    /// Похоже на Млечный Путь (в пределах ~30 кпк).
    pub fn milky_way(n: usize) -> Self {
        DiskParams { n_disk: n * 5 / 10, n_bulge: n / 10, n_halo: n * 4 / 10, m_disk: 5.0, r_disk: 2.6, z_disk: 0.3,
                     m_bulge: 1.0, a_bulge: 0.6, m_halo: 40.0, a_halo: 20.0 }
    }
}

fn plummer_mass(m: f64, a: f64, r: f64) -> f64 { m * r.powi(3) / (r * r + a * a).powf(1.5) }

/// Дисковая галактика: гало + балдж + экспоненциальный диск на круговых орбитах с дисперсией.
pub fn disk_galaxy(rng: &mut Rng, p: &DiskParams) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>, Vec<Kind>) {
    let (mut pos, mut vel, mut mass, mut kind) = plummer(rng, p.n_halo, p.m_halo, p.a_halo, Kind::Halo);
    let (bp, bv, bm, bk) = plummer(rng, p.n_bulge, p.m_bulge, p.a_bulge, Kind::Bulge);
    pos.extend(bp); vel.extend(bv); mass.extend(bm); kind.extend(bk);
    for _ in 0..p.n_disk {
        // экспоненциальный профиль Σ ∝ e^{−R/Rd}: R из гамма(2) распределения
        let r = -p.r_disk * (rng.f64().max(1e-12) * rng.f64().max(1e-12)).ln();
        let r = r.min(8.0 * p.r_disk);
        let ph = rng.uniform(0.0, 2.0 * PI);
        let z = p.z_disk * (2.0 * rng.f64() - 1.0).atanh() * 0.5;
        let m_enc = plummer_mass(p.m_halo, p.a_halo, r) + plummer_mass(p.m_bulge, p.a_bulge, r)
            + p.m_disk * (1.0 - (1.0 + r / p.r_disk) * (-r / p.r_disk).exp());
        let vc = (G_GAL * m_enc / r.max(0.05)).sqrt();
        let sigma = 0.1 * vc;
        let tang = Vec3::new(-ph.sin(), ph.cos(), 0.0);
        let rad = Vec3::new(ph.cos(), ph.sin(), 0.0);
        pos.push(Vec3::new(r * ph.cos(), r * ph.sin(), z));
        vel.push(tang * (vc + sigma * 0.5 * rng.normal()) + rad * (sigma * rng.normal()) + Vec3::new(0.0, 0.0, 0.5 * sigma * rng.normal()));
        mass.push(p.m_disk / p.n_disk as f64);
        kind.push(Kind::Disk);
    }
    (pos, vel, mass, kind)
}

pub const PRESETS: &[&str] = &["milky_way", "collision", "plummer", "globular"];

pub fn preset(name: &str, n: usize, seed: u64) -> Result<Galaxy, String> {
    let mut rng = Rng::new(seed);
    Ok(match name {
        "milky_way" => {
            let (p, v, m, k) = disk_galaxy(&mut rng, &DiskParams::milky_way(n));
            Galaxy::new(p, v, m, k, 0.1)
        }
        "collision" => {
            // две галактики на почти параболической встрече (как Млечный Путь и Андромеда)
            let (mut p1, mut v1, mut m1, mut k1) = disk_galaxy(&mut rng, &DiskParams::milky_way(n / 2));
            let (p2, v2, m2, k2) = disk_galaxy(&mut rng, &DiskParams::milky_way(n / 2));
            let tilt = |q: Vec3| { let (s, c) = (60f64.to_radians().sin(), 60f64.to_radians().cos()); Vec3::new(q.x, c * q.y - s * q.z, s * q.y + c * q.z) };
            let mtot: f64 = m1.iter().sum::<f64>() * 2.0;
            let d = 70.0;
            let b = 12.0;
            let v_par = (2.0 * G_GAL * mtot / d).sqrt() * 0.9;
            let off = Vec3::new(d / 2.0, b / 2.0, 0.0);
            let vo = Vec3::new(-v_par / 2.0, 0.0, 0.0);
            for i in 0..p1.len() { p1[i] = p1[i] - off; v1[i] = v1[i] - vo; }
            for i in 0..p2.len() { p1.push(tilt(p2[i]) + off); v1.push(tilt(v2[i]) + vo); }
            m1.extend(m2);
            k1.extend(k2);
            Galaxy::new(p1, v1, m1, k1, 0.12)
        }
        "plummer" => {
            let (p, v, m, k) = plummer(&mut rng, n, 1.0, 1.0, Kind::Star);
            Galaxy::new(p, v, m, k, 0.02)
        }
        "globular" => {
            // шаровое скопление: 10⁵·… M☉ (~ M13), a = 3 пк
            let (p, v, m, k) = plummer(&mut rng, n, 6.0e-5, 0.003, Kind::Star);
            Galaxy::new(p, v, m, k, 0.0002)
        }
        _ => return Err(format!("неизвестный пресет галактики '{name}', доступны: {PRESETS:?}")),
    })
}

/// Время пересечения для пламмеровской сферы: t_cr = (G M)^{-1/2} · a^{3/2} · ~
pub fn crossing_time(m: f64, a: f64) -> f64 { (a.powi(3) / (G_GAL * m)).sqrt() }
