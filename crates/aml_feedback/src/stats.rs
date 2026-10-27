//! Small statistics helpers (no external crates).

/// Wilson score interval for a (possibly weighted) proportion. `n` is the effective sample size.
pub fn wilson(pos: f64, n: f64, z: f64) -> (f64, f64, f64) {
    if n <= 0.0 {
        return (0.0, 0.0, 1.0);
    }
    let p = (pos / n).clamp(0.0, 1.0);
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).max(0.0)).sqrt() / denom;
    (p, (centre - half).clamp(0.0, 1.0), (centre + half).clamp(0.0, 1.0))
}

pub fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    (p / (1.0 - p)).ln()
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wilson_basic() {
        let (p, lo, hi) = wilson(50.0, 100.0, 1.96);
        assert!((p - 0.5).abs() < 1e-12);
        assert!(lo < 0.5 && hi > 0.5 && lo > 0.39 && hi < 0.61);
        let (_, lo0, hi0) = wilson(0.0, 0.0, 1.96);
        assert_eq!((lo0, hi0), (0.0, 1.0));
        let (_, lo1, hi1) = wilson(10.0, 10.0, 1.96);
        assert!(hi1 <= 1.0 && lo1 > 0.65);
    }
    #[test]
    fn logit_roundtrip() {
        for p in [0.01, 0.3, 0.5, 0.9] {
            assert!((sigmoid(logit(p)) - p).abs() < 1e-9);
        }
    }
}
