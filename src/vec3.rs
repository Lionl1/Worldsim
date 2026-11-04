use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn norm2(self) -> f64 {
        self.dot(self)
    }
    pub fn norm(self) -> f64 {
        self.norm2().sqrt()
    }
    pub fn unit(self) -> Vec3 {
        let n = self.norm();
        if n == 0.0 { self } else { self / n }
    }
}

impl Add for Vec3 { type Output = Vec3; fn add(self, o: Vec3) -> Vec3 { Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z) } }
impl Sub for Vec3 { type Output = Vec3; fn sub(self, o: Vec3) -> Vec3 { Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z) } }
impl Mul<f64> for Vec3 { type Output = Vec3; fn mul(self, s: f64) -> Vec3 { Vec3::new(self.x * s, self.y * s, self.z * s) } }
impl Mul<Vec3> for f64 { type Output = Vec3; fn mul(self, v: Vec3) -> Vec3 { v * self } }
impl Div<f64> for Vec3 { type Output = Vec3; fn div(self, s: f64) -> Vec3 { Vec3::new(self.x / s, self.y / s, self.z / s) } }
impl Neg for Vec3 { type Output = Vec3; fn neg(self) -> Vec3 { Vec3::new(-self.x, -self.y, -self.z) } }
impl AddAssign for Vec3 { fn add_assign(&mut self, o: Vec3) { *self = *self + o; } }
impl SubAssign for Vec3 { fn sub_assign(&mut self, o: Vec3) { *self = *self - o; } }
