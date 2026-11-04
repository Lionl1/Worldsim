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

    pub fn get(&self, k: &str) -> Option<&J> {
        if let J::Obj(v) = self { v.iter().find(|(n, _)| n == k).map(|(_, x)| x) } else { None }
    }
    pub fn as_f64(&self) -> Option<f64> { if let J::Num(x) = self { Some(*x) } else { None } }
    pub fn as_str(&self) -> Option<&str> { if let J::Str(x) = self { Some(x) } else { None } }
    pub fn as_arr(&self) -> Option<&[J]> { if let J::Arr(x) = self { Some(x) } else { None } }
    pub fn as_obj(&self) -> Option<&[(String, J)]> { if let J::Obj(x) = self { Some(x) } else { None } }

    /// Разбор JSON (описание миссии): без зависимостей, ошибки — с позицией.
    pub fn parse(s: &str) -> Result<J, String> {
        let b = s.as_bytes();
        let mut i = 0;
        let v = parse_value(b, &mut i, 0)?;
        skip_ws(b, &mut i);
        if i != b.len() { return Err(format!("JSON: лишний текст на позиции {i}")); }
        Ok(v)
    }
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

fn skip_ws(b: &[u8], i: &mut usize) { while *i < b.len() && (b[*i] as char).is_ascii_whitespace() { *i += 1; } }

fn parse_value(b: &[u8], i: &mut usize, depth: usize) -> Result<J, String> {
    if depth > 64 { return Err("JSON: слишком большая вложенность".into()); }
    skip_ws(b, i);
    let err = |i: usize, m: &str| format!("JSON: {m} на позиции {i}");
    match b.get(*i) {
        None => Err(err(*i, "неожиданный конец")),
        Some(b'{') => {
            *i += 1;
            let mut out = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b'}') { *i += 1; return Ok(J::Obj(out)); }
            loop {
                skip_ws(b, i);
                let J::Str(k) = parse_value(b, i, depth + 1)? else { return Err(err(*i, "ключ должен быть строкой")); };
                skip_ws(b, i);
                if b.get(*i) != Some(&b':') { return Err(err(*i, "ожидалось ':'")); }
                *i += 1;
                out.push((k, parse_value(b, i, depth + 1)?));
                skip_ws(b, i);
                match b.get(*i) { Some(b',') => *i += 1, Some(b'}') => { *i += 1; return Ok(J::Obj(out)); } _ => return Err(err(*i, "ожидалось ',' или '}'")) }
            }
        }
        Some(b'[') => {
            *i += 1;
            let mut out = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b']') { *i += 1; return Ok(J::Arr(out)); }
            loop {
                out.push(parse_value(b, i, depth + 1)?);
                skip_ws(b, i);
                match b.get(*i) { Some(b',') => *i += 1, Some(b']') => { *i += 1; return Ok(J::Arr(out)); } _ => return Err(err(*i, "ожидалось ',' или ']'")) }
            }
        }
        Some(b'"') => {
            *i += 1;
            let mut out = Vec::new();
            loop {
                match b.get(*i) {
                    None => return Err(err(*i, "незакрытая строка")),
                    Some(b'"') => { *i += 1; break; }
                    Some(b'\\') => {
                        *i += 1;
                        match b.get(*i) {
                            Some(b'"') => out.push(b'"'), Some(b'\\') => out.push(b'\\'), Some(b'/') => out.push(b'/'),
                            Some(b'n') => out.push(b'\n'), Some(b't') => out.push(b'\t'), Some(b'r') => out.push(b'\r'),
                            Some(b'b') => out.push(8), Some(b'f') => out.push(12),
                            Some(b'u') => {
                                let h = b.get(*i + 1..*i + 5).and_then(|x| std::str::from_utf8(x).ok()).and_then(|x| u32::from_str_radix(x, 16).ok()).ok_or_else(|| err(*i, "плохой \\u"))?;
                                let c = char::from_u32(h).unwrap_or('\u{FFFD}');
                                let mut t = [0u8; 4];
                                out.extend_from_slice(c.encode_utf8(&mut t).as_bytes());
                                *i += 4;
                            }
                            _ => return Err(err(*i, "плохая escape-последовательность")),
                        }
                        *i += 1;
                    }
                    Some(c) => { out.push(*c); *i += 1; }
                }
            }
            String::from_utf8(out).map(J::Str).map_err(|_| err(*i, "не UTF-8"))
        }
        Some(b't') if b[*i..].starts_with(b"true") => { *i += 4; Ok(J::Bool(true)) }
        Some(b'f') if b[*i..].starts_with(b"false") => { *i += 5; Ok(J::Bool(false)) }
        Some(b'n') if b[*i..].starts_with(b"null") => { *i += 4; Ok(J::Null) }
        Some(_) => {
            let st = *i;
            while *i < b.len() && matches!(b[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') { *i += 1; }
            std::str::from_utf8(&b[st..*i]).ok().and_then(|x| x.parse().ok()).map(J::Num).ok_or_else(|| err(st, "ожидалось значение"))
        }
    }
}
