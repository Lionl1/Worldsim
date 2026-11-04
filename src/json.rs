//! Минимальный построитель JSON (без зависимостей).

use std::fmt::Write;

#[derive(Clone, Debug)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn obj() -> J { J::Obj(Vec::new()) }
    pub fn set(mut self, k: &str, v: impl Into<J>) -> J {
        if let J::Obj(ref mut m) = self { m.push((k.to_string(), v.into())); }
        self
    }
    pub fn render(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }
    fn write(&self, s: &mut String) {
        match self {
            J::Null => s.push_str("null"),
            J::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            J::Num(x) => {
                if x.is_finite() {
                    if x.fract() == 0.0 && x.abs() < 1e15 { let _ = write!(s, "{}", *x as i64); } else { let _ = write!(s, "{:.7e}", x); }
                } else { s.push_str("null") }
            }
            J::Str(t) => esc(t, s),
            J::Arr(v) => {
                s.push('[');
                for (i, x) in v.iter().enumerate() { if i > 0 { s.push(','); } x.write(s); }
                s.push(']');
            }
            J::Obj(m) => {
                s.push('{');
                for (i, (k, v)) in m.iter().enumerate() { if i > 0 { s.push(','); } esc(k, s); s.push(':'); v.write(s); }
                s.push('}');
            }
        }
    }
}

fn esc(t: &str, s: &mut String) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            c if (c as u32) < 0x20 => { let _ = write!(s, "\\u{:04x}", c as u32); }
            c => s.push(c),
        }
    }
    s.push('"');
}

impl From<f64> for J { fn from(x: f64) -> J { J::Num(x) } }
impl From<usize> for J { fn from(x: usize) -> J { J::Num(x as f64) } }
impl From<u64> for J { fn from(x: u64) -> J { J::Num(x as f64) } }
impl From<bool> for J { fn from(x: bool) -> J { J::Bool(x) } }
impl From<&str> for J { fn from(x: &str) -> J { J::Str(x.to_string()) } }
impl From<String> for J { fn from(x: String) -> J { J::Str(x) } }
impl From<Vec<f64>> for J { fn from(v: Vec<f64>) -> J { J::Arr(v.into_iter().map(J::Num).collect()) } }
impl From<&[f64]> for J { fn from(v: &[f64]) -> J { J::Arr(v.iter().map(|x| J::Num(*x)).collect()) } }
impl From<Vec<J>> for J { fn from(v: Vec<J>) -> J { J::Arr(v) } }
impl From<Vec<String>> for J { fn from(v: Vec<String>) -> J { J::Arr(v.into_iter().map(J::Str).collect()) } }
impl<T: Into<J>> From<Option<T>> for J { fn from(v: Option<T>) -> J { v.map(Into::into).unwrap_or(J::Null) } }
