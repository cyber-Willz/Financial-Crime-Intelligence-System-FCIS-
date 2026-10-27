//! Learned alert prioritisation.
//!
//! The rule engine in [`crate::screening`] produces a hand-set score. This module adds a
//! **learned match-probability model** on top of the same evidence, trained on what your analysts
//! actually decided (false positive vs confirmed match), so the review queue can be ordered by the
//! likelihood of a true match instead of by a fixed formula.
//!
//! # What it does and does not do
//!
//! * It **only orders** alerts. It never creates, suppresses, dismisses, confirms, upgrades or
//!   downgrades an alert, and it never changes a [`crate::screening::HitLevel`]. Every alert still
//!   needs a human disposition with the same role rules as before. (A model that auto-dismissed would
//!   have to be validated for *missed matches*, which this data cannot measure: only alerts the rules
//!   already raised are ever labelled.)
//! * It cannot find a match the rules did not raise. It reduces review *effort*, not missed-match risk.
//! * Alerts that are high-level or carry a strong identifier match are always in the
//!   [`PriorityBand::Critical`] band, whatever the model says.
//!
//! # Model
//!
//! Plain logistic regression on [`N_FEATURES`] interpretable, scaled features ([`MatchFeatures`]).
//! Interpretable on purpose: every priority can be explained as a sum of per-feature contributions
//! ([`LogisticModel::explain`]), which is what a model-risk reviewer or regulator will ask for.
//!
//! Real labelled data is scarce (confirmed matches are rare). So training is a **MAP fit with the
//! hand-set scorecard as the prior**: coefficients are pulled toward [`LogisticModel::prior`] with a
//! ridge penalty whose strength is chosen by cross-validation. With few labels the fit stays near the
//! scorecard; with many it follows the data. Safety-critical coefficients are held to the sign common
//! sense dictates (a strong identifier match can never *lower* the priority, a conflicting date of
//! birth can never *raise* it); a violating coefficient is frozen at zero and the model refitted.
//!
//! # Validation and activation
//!
//! [`train`] runs a nested, **grouped** (by customer) and positive-spread k-fold cross-validation, and
//! reports: AUC of the model, the hand-set scorecard prior and the plain rule score; a bootstrap 95%
//! interval for the model-minus-rule AUC; average precision, Brier score, log-loss, reliability bins, and
//! the share of the queue an analyst must read to reach 95% / 100% of the true matches. A model may be
//! **activated** only if every [`ActivationGates`] check passes. Until then the engine runs in
//! `Shadow` mode: it records the priority at alert creation, so [`shadow_eval`] can later measure the
//! model prospectively on alerts that were resolved *after* it was scored.
//!
//! # What is not established
//!
//! * The prior (scorecard) coefficients are hand-set expert guesses, not fitted to real data.
//! * Nothing here is validated on real screening data until you run it on yours.
//! * Labels are analyst decisions, so the model inherits analyst habits and errors. Alerts reviewed
//!   in priority order stay unbiased only while *all* alerts are eventually reviewed.

use crate::calibrate::auc;
use crate::error::{Error, Result};
use crate::model::{AliasStrength, DobRel, SubjectKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeMap;

// ----------------------------------------------------------------------------------------
// Features
// ----------------------------------------------------------------------------------------

/// Structured evidence behind one screening hit. Captured when the hit is produced and stored on the
/// alert, so training later sees exactly what the analyst saw (not a re-computation on newer lists).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MatchFeatures {
    /// Final rule-engine score (name score plus DOB/nationality/identifier adjustments), 0..1.
    pub rule_score: f64,
    /// Alias-discounted best name score, 0..1.
    pub name_score: f64,
    /// Name tokens that matched at >= 0.88 similarity.
    pub strong_tokens: u32,
    /// Tokens in the screened name variant / in the listed name variant.
    pub subject_tokens: u32,
    pub entry_tokens: u32,
    /// The matched listed name is a weak/former alias.
    pub weak_alias: bool,
    /// The match came from one of the customer's own aliases rather than their primary name.
    pub subject_alias_used: bool,
    /// Same tokens in any order, no fuzzing.
    pub exact_name: bool,
    pub dob: DobCode,
    pub nationality: NatCode,
    pub country_assoc: bool,
    pub id: IdCode,
    /// The listed party is an entity/vessel/aircraft rather than an individual.
    pub entry_non_individual: bool,
    /// ln(N / df) of the rarest / mean name token shared exactly with the listed name, over the index
    /// size N. A match on a rare token is far more specific than one on a common surname.
    pub rarest_shared_idf: f64,
    pub mean_shared_idf: f64,
    /// Other hits raised for the same customer screening run (a very common name hits many entries).
    pub competing_hits: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DobCode {
    #[default]
    Unknown,
    Exact,
    YearOnly,
    NearMiss,
    Conflict,
}

impl From<DobRel> for DobCode {
    fn from(r: DobRel) -> Self {
        match r {
            DobRel::Exact => Self::Exact,
            DobRel::YearOnly => Self::YearOnly,
            DobRel::NearMiss => Self::NearMiss,
            DobRel::Conflict => Self::Conflict,
            DobRel::Unknown => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NatCode {
    #[default]
    Unknown,
    Match,
    Differs,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdCode {
    #[default]
    None,
    /// Same identifier of a compatible kind/country.
    Strong,
    /// Same number, but kind or issuing country differs.
    Weak,
}

pub const N_FEATURES: usize = 21;

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "rule_score",
    "name_score",
    "strong_tokens",
    "subject_tokens",
    "token_count_gap",
    "weak_alias",
    "subject_alias_used",
    "exact_name",
    "dob_exact",
    "dob_year_only",
    "dob_near_miss",
    "dob_conflict",
    "nationality_match",
    "nationality_differs",
    "country_assoc",
    "id_strong",
    "id_weak",
    "entry_non_individual",
    "rarest_shared_idf",
    "mean_shared_idf",
    "competing_hits",
];

impl MatchFeatures {
    /// The scaled design vector (every entry roughly in 0..1, so a single ridge strength is sensible).
    pub fn vector(&self) -> [f64; N_FEATURES] {
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        let gap = (self.subject_tokens as f64 - self.entry_tokens as f64).abs();
        [
            self.rule_score.clamp(0.0, 1.0),
            self.name_score.clamp(0.0, 1.0),
            (self.strong_tokens as f64).min(4.0) / 4.0,
            (self.subject_tokens as f64).min(6.0) / 6.0,
            gap.min(4.0) / 4.0,
            b(self.weak_alias),
            b(self.subject_alias_used),
            b(self.exact_name),
            b(self.dob == DobCode::Exact),
            b(self.dob == DobCode::YearOnly),
            b(self.dob == DobCode::NearMiss),
            b(self.dob == DobCode::Conflict),
            b(self.nationality == NatCode::Match),
            b(self.nationality == NatCode::Differs),
            b(self.country_assoc),
            b(self.id == IdCode::Strong),
            b(self.id == IdCode::Weak),
            b(self.entry_non_individual),
            (self.rarest_shared_idf / 10.0).clamp(0.0, 1.5),
            (self.mean_shared_idf / 10.0).clamp(0.0, 1.5),
            (1.0 + self.competing_hits as f64).ln().min(3.0) / 3.0,
        ]
    }

    pub fn alias_strength_weak(s: AliasStrength) -> bool {
        matches!(s, AliasStrength::Weak)
    }

    pub fn non_individual(k: SubjectKind) -> bool {
        k != SubjectKind::Individual
    }
}

// ----------------------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------------------

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// log(1 + e^z), stable.
fn softplus(z: f64) -> f64 {
    if z > 30.0 {
        z
    } else if z < -30.0 {
        z.exp()
    } else {
        (1.0 + z.exp()).ln()
    }
}

/// Sign a coefficient must keep (safety constraint), by feature name.
fn required_sign(name: &str) -> Option<f64> {
    match name {
        "rule_score" | "name_score" | "strong_tokens" | "exact_name" | "dob_exact" | "nationality_match" | "id_strong" | "id_weak" => {
            Some(1.0)
        }
        "dob_conflict" | "nationality_differs" | "weak_alias" => Some(-1.0),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub examples: usize,
    pub positives: usize,
    pub lambda: f64,
    pub cv_auc: Option<f64>,
    pub rule_auc: Option<f64>,
    pub trained_on: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogisticModel {
    pub version: String,
    pub bias: f64,
    pub weights: Vec<f64>,
    /// Features frozen at zero because the data pushed them against their required sign.
    pub frozen: Vec<String>,
    pub training: Option<TrainingSummary>,
}

pub const PRIOR_VERSION: &str = "prior-v1";

impl LogisticModel {
    /// The hand-set scorecard in logistic form. **Not fitted to data**: reasonable expert guesses so that
    /// a name-only match on a common name scores low, a corroborated or identifier match scores high.
    pub fn prior() -> Self {
        let mut w = vec![0.0; N_FEATURES];
        let mut set = |n: &str, v: f64| {
            let i = FEATURE_NAMES.iter().position(|x| *x == n).expect("feature");
            w[i] = v;
        };
        set("rule_score", 4.0);
        set("name_score", 6.0);
        set("strong_tokens", 1.6);
        set("token_count_gap", -0.5);
        set("weak_alias", -0.7);
        set("exact_name", 0.5);
        set("dob_exact", 3.0);
        set("dob_year_only", 1.0);
        set("dob_near_miss", -0.5);
        set("dob_conflict", -2.0);
        set("nationality_match", 0.8);
        set("nationality_differs", -0.8);
        set("country_assoc", 0.3);
        set("id_strong", 4.0);
        set("id_weak", 1.5);
        set("rarest_shared_idf", 2.0);
        set("mean_shared_idf", 1.0);
        set("competing_hits", -1.0);
        Self { version: PRIOR_VERSION.into(), bias: -9.0, weights: w, frozen: vec![], training: None }
    }

    pub fn is_prior(&self) -> bool {
        self.training.is_none()
    }

    pub fn logit_vec(&self, x: &[f64; N_FEATURES]) -> f64 {
        self.bias + self.weights.iter().zip(x.iter()).map(|(w, v)| w * v).sum::<f64>()
    }

    pub fn predict_vec(&self, x: &[f64; N_FEATURES]) -> f64 {
        sigmoid(self.logit_vec(x))
    }

    /// Match-likelihood in 0..1. Treat it as a probability only for a trained model whose reliability bins you have
    /// checked; for the hand-set prior it is a ranking score.
    pub fn predict(&self, f: &MatchFeatures) -> f64 {
        self.predict_vec(&f.vector())
    }

    /// Per-feature contributions to the logit, largest magnitude first (zero contributions omitted).
    pub fn explain(&self, f: &MatchFeatures) -> Vec<(&'static str, f64)> {
        let x = f.vector();
        let mut v: Vec<(&'static str, f64)> =
            FEATURE_NAMES.iter().zip(self.weights.iter().zip(x.iter())).map(|(n, (w, x))| (*n, w * x)).filter(|(_, c)| c.abs() > 1e-9).collect();
        v.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
        v
    }

    /// Stable fingerprint of the coefficients, recorded in the audit log.
    pub fn fingerprint(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.version.as_bytes());
        h.update(self.bias.to_le_bytes());
        for w in &self.weights {
            h.update(w.to_le_bytes());
        }
        h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    pub fn validate(&self) -> Result<()> {
        if self.weights.len() != N_FEATURES {
            return Err(Error::Invalid(format!("model has {} weights, expected {N_FEATURES}", self.weights.len())));
        }
        if !self.bias.is_finite() || self.weights.iter().any(|w| !w.is_finite()) {
            return Err(Error::Invalid("model has non-finite coefficients".into()));
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------------------
// Priority bands and queue ordering
// ----------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorityBand {
    /// High-level alert or a strong identifier match: always read first, never reordered by the model.
    Critical,
    Standard,
    /// Review-level, no identifier, no exact date of birth, and a model priority under the configured
    /// cut-off. **Still open and still needs a human decision**; the band only affects reading order.
    Low,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlMode {
    /// No priority is computed.
    Off,
    /// Priority is computed and stored, but the queue is still ordered by the rule score.
    Shadow,
    /// The queue is ordered by the validated model's priority.
    Active,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MlConfig {
    pub mode: MlMode,
    /// Review-level alerts with priority below this go to the Low band. `0` disables the Low band
    /// (the default): set it from a validated training report, never by guess.
    pub low_band_max_priority: f64,
}

impl Default for MlConfig {
    fn default() -> Self {
        Self { mode: MlMode::Shadow, low_band_max_priority: 0.0 }
    }
}

pub fn band(is_high_level: bool, f: &MatchFeatures, priority: Option<f64>, cfg: &MlConfig) -> PriorityBand {
    if is_high_level || f.id == IdCode::Strong {
        return PriorityBand::Critical;
    }
    match priority {
        Some(p) if cfg.low_band_max_priority > 0.0 && p < cfg.low_band_max_priority && f.id == IdCode::None && f.dob != DobCode::Exact => {
            PriorityBand::Low
        }
        _ => PriorityBand::Standard,
    }
}

// ----------------------------------------------------------------------------------------
// Training data
// ----------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Example {
    pub x: [f64; N_FEATURES],
    /// `true` = confirmed match.
    pub y: bool,
    /// Customer id: all examples of one customer land in the same CV fold.
    pub group: u64,
    pub rule_score: f64,
}

impl Example {
    pub fn new(f: &MatchFeatures, y: bool, group: u64) -> Self {
        Self { x: f.vector(), y, group, rule_score: f.rule_score }
    }
}

// ----------------------------------------------------------------------------------------
// Fitting (MAP logistic regression, Newton / IRLS with a ridge pull toward the prior)
// ----------------------------------------------------------------------------------------

const P: usize = N_FEATURES + 1; // bias first

struct FitSpec<'a> {
    prior: &'a LogisticModel,
    lambda: f64,
    frozen: [bool; N_FEATURES],
}

fn objective(theta: &[f64; P], theta0: &[f64; P], lam: &[f64; P], ex: &[&Example], frozen: &[bool; N_FEATURES]) -> f64 {
    let mut o = 0.0;
    for e in ex {
        let z = zof(theta, &e.x, frozen);
        o += softplus(z) - if e.y { z } else { 0.0 };
    }
    for j in 0..P {
        let d = theta[j] - theta0[j];
        o += 0.5 * lam[j] * d * d;
    }
    o
}

fn zof(theta: &[f64; P], x: &[f64; N_FEATURES], frozen: &[bool; N_FEATURES]) -> f64 {
    let mut z = theta[0];
    for j in 0..N_FEATURES {
        if !frozen[j] {
            z += theta[j + 1] * x[j];
        }
    }
    z
}

fn fit(ex: &[&Example], spec: &FitSpec) -> [f64; P] {
    let mut theta0 = [0.0; P];
    theta0[0] = spec.prior.bias;
    for j in 0..N_FEATURES {
        theta0[j + 1] = if spec.frozen[j] { 0.0 } else { spec.prior.weights[j] };
    }
    let mut lam = [spec.lambda; P];
    lam[0] = 1e-6; // the intercept must be free to follow the base rate
    let mut theta = theta0;
    let mut obj = objective(&theta, &theta0, &lam, ex, &spec.frozen);
    for _ in 0..100 {
        let mut g = nalgebra::DVector::<f64>::zeros(P);
        let mut h = nalgebra::DMatrix::<f64>::zeros(P, P);
        for e in ex {
            let p = sigmoid(zof(&theta, &e.x, &spec.frozen));
            let w = (p * (1.0 - p)).max(1e-9);
            let r = p - if e.y { 1.0 } else { 0.0 };
            let mut xt = [0.0; P];
            xt[0] = 1.0;
            for j in 0..N_FEATURES {
                xt[j + 1] = if spec.frozen[j] { 0.0 } else { e.x[j] };
            }
            for a in 0..P {
                g[a] += r * xt[a];
                for b in a..P {
                    h[(a, b)] += w * xt[a] * xt[b];
                }
            }
        }
        for a in 0..P {
            g[a] += lam[a] * (theta[a] - theta0[a]);
            h[(a, a)] += lam[a];
            for b in 0..a {
                h[(a, b)] = h[(b, a)];
            }
        }
        let step = match h.clone().cholesky() {
            Some(c) => c.solve(&g),
            None => match h.lu().solve(&g) {
                Some(s) => s,
                None => break,
            },
        };
        // backtracking line search on the penalised negative log-likelihood
        let mut t = 1.0;
        let mut accepted = false;
        while t > 1e-6 {
            let mut cand = theta;
            for a in 0..P {
                cand[a] -= t * step[a];
            }
            let o = objective(&cand, &theta0, &lam, ex, &spec.frozen);
            if o <= obj + 1e-12 {
                let delta = (0..P).map(|a| (cand[a] - theta[a]).abs()).fold(0.0, f64::max);
                theta = cand;
                obj = o;
                accepted = true;
                if delta < 1e-9 {
                    return theta;
                }
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    theta
}

fn model_from(theta: &[f64; P], frozen: &[bool; N_FEATURES], version: &str) -> LogisticModel {
    LogisticModel {
        version: version.into(),
        bias: theta[0],
        weights: (0..N_FEATURES).map(|j| if frozen[j] { 0.0 } else { theta[j + 1] }).collect(),
        frozen: (0..N_FEATURES).filter(|j| frozen[*j]).map(|j| FEATURE_NAMES[j].to_string()).collect(),
        training: None,
    }
}

/// Fit with the sign constraints: any coefficient on the wrong side is frozen at zero and the model refitted.
fn fit_constrained(ex: &[&Example], prior: &LogisticModel, lambda: f64) -> ([f64; P], [bool; N_FEATURES]) {
    let mut frozen = [false; N_FEATURES];
    let mut theta = fit(ex, &FitSpec { prior, lambda, frozen });
    for _ in 0..4 {
        let mut changed = false;
        for j in 0..N_FEATURES {
            if frozen[j] {
                continue;
            }
            if let Some(s) = required_sign(FEATURE_NAMES[j]) {
                if theta[j + 1] * s < -1e-12 {
                    frozen[j] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        theta = fit(ex, &FitSpec { prior, lambda, frozen });
    }
    (theta, frozen)
}

// ----------------------------------------------------------------------------------------
// Deterministic randomness and folds
// ----------------------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// Fold index per example: customers are kept whole, and customers with a confirmed match are dealt out
/// first so the rare positives spread over the folds.
fn grouped_folds(ex: &[&Example], k: usize, seed: u64) -> Vec<usize> {
    let mut groups: BTreeMap<u64, bool> = BTreeMap::new();
    for e in ex {
        let g = groups.entry(e.group).or_insert(false);
        *g |= e.y;
    }
    let mut pos: Vec<u64> = groups.iter().filter(|(_, p)| **p).map(|(g, _)| *g).collect();
    let mut neg: Vec<u64> = groups.iter().filter(|(_, p)| !**p).map(|(g, _)| *g).collect();
    let mut rng = Rng(seed);
    rng.shuffle(&mut pos);
    rng.shuffle(&mut neg);
    let mut fold_of: BTreeMap<u64, usize> = BTreeMap::new();
    for (i, g) in pos.iter().enumerate() {
        fold_of.insert(*g, i % k);
    }
    // negatives continue the round-robin from where positives stopped, keeping fold sizes even
    let start = pos.len();
    for (i, g) in neg.iter().enumerate() {
        fold_of.insert(*g, (start + i) % k);
    }
    ex.iter().map(|e| fold_of[&e.group]).collect()
}

// ----------------------------------------------------------------------------------------
// Metrics
// ----------------------------------------------------------------------------------------

pub fn average_precision(s: &[(bool, f64)]) -> Option<f64> {
    let p = s.iter().filter(|(l, _)| *l).count();
    if p == 0 {
        return None;
    }
    let mut v = s.to_vec();
    // ties: negatives first (pessimistic)
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
    let (mut tp, mut ap) = (0.0, 0.0);
    for (i, (l, _)) in v.iter().enumerate() {
        if *l {
            tp += 1.0;
            ap += tp / (i as f64 + 1.0);
        }
    }
    Some(ap / p as f64)
}

pub fn brier(s: &[(bool, f64)]) -> f64 {
    s.iter().map(|(l, p)| (p - if *l { 1.0 } else { 0.0 }).powi(2)).sum::<f64>() / s.len().max(1) as f64
}

pub fn log_loss(s: &[(bool, f64)]) -> f64 {
    let eps = 1e-12;
    s.iter().map(|(l, p)| -(if *l { p.max(eps).ln() } else { (1.0 - p).max(eps).ln() })).sum::<f64>() / s.len().max(1) as f64
}

/// Fraction of the queue (read from the top in score order) needed to reach `recall` of all true
/// matches. Ties are resolved pessimistically (negatives first), so a model cannot look good by tying.
pub fn queue_fraction_at_recall(s: &[(bool, f64)], recall: f64) -> Option<f64> {
    let p = s.iter().filter(|(l, _)| *l).count();
    if p == 0 || s.is_empty() {
        return None;
    }
    let need = ((recall * p as f64).ceil() as usize).max(1);
    let mut v = s.to_vec();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
    let mut found = 0;
    for (i, (l, _)) in v.iter().enumerate() {
        if *l {
            found += 1;
            if found >= need {
                return Some((i + 1) as f64 / v.len() as f64);
            }
        }
    }
    None
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBin {
    pub count: usize,
    pub mean_predicted: f64,
    pub observed_rate: f64,
}

pub fn reliability(s: &[(bool, f64)], bins: usize) -> Vec<ReliabilityBin> {
    let mut v = s.to_vec();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return vec![];
    }
    let bins = bins.min(n).max(1);
    (0..bins)
        .map(|b| {
            let lo = b * n / bins;
            let hi = (b + 1) * n / bins;
            let c = &v[lo..hi];
            ReliabilityBin {
                count: c.len(),
                mean_predicted: c.iter().map(|x| x.1).sum::<f64>() / c.len() as f64,
                observed_rate: c.iter().filter(|x| x.0).count() as f64 / c.len() as f64,
            }
        })
        .collect()
}

/// 95% bootstrap interval (percentile) of AUC(a) - AUC(b) on the same labelled items.
fn bootstrap_auc_delta(items: &[(bool, f64, f64)], rounds: usize, seed: u64) -> Option<(f64, f64)> {
    if rounds == 0 || items.is_empty() {
        return None;
    }
    let mut rng = Rng(seed ^ 0xA5A5);
    let mut deltas = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let mut a = Vec::with_capacity(items.len());
        let mut b = Vec::with_capacity(items.len());
        for _ in 0..items.len() {
            let it = items[rng.below(items.len())];
            a.push((it.0, it.1));
            b.push((it.0, it.2));
        }
        if let (Some(x), Some(y)) = (auc(&a), auc(&b)) {
            deltas.push(x - y);
        }
    }
    if deltas.len() < rounds / 2 {
        return None;
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let q = |p: f64| deltas[((deltas.len() - 1) as f64 * p).round() as usize];
    Some((q(0.025), q(0.975)))
}

// ----------------------------------------------------------------------------------------
// Training and evaluation
// ----------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationGates {
    pub min_examples: usize,
    pub min_positives: usize,
    pub min_negatives: usize,
    /// The lower end of the bootstrap 95% interval of (model AUC - rule-score AUC) must exceed this.
    pub min_auc_gain_lower_bound: f64,
    /// Queue fraction needed to reach 100% of the true matches may be at most this much worse than the
    /// rule score's (0 = not worse).
    pub max_workload_regression: f64,
}

impl Default for ActivationGates {
    fn default() -> Self {
        Self { min_examples: 200, min_positives: 20, min_negatives: 20, min_auc_gain_lower_bound: 0.0, max_workload_regression: 0.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainOptions {
    pub folds: usize,
    pub seed: u64,
    /// Candidate ridge strengths (pull toward the scorecard); chosen by inner cross-validated log-loss.
    pub lambda_grid: Vec<f64>,
    pub bootstrap_rounds: usize,
    pub gates: ActivationGates,
    /// Label for the resulting model's version string, e.g. a date.
    pub label: String,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self { folds: 5, seed: 20260101, lambda_grid: vec![0.3, 3.0, 30.0, 300.0], bootstrap_rounds: 1000, gates: ActivationGates::default(), label: "trained".into() }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gate {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CvMetrics {
    pub auc: Option<f64>,
    pub average_precision: Option<f64>,
    pub brier: f64,
    pub log_loss: f64,
    pub queue_at_95_recall: Option<f64>,
    pub queue_at_100_recall: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainReport {
    pub examples: usize,
    pub positives: usize,
    pub negatives: usize,
    pub customers: usize,
    pub folds: usize,
    pub lambda: f64,
    /// Out-of-fold (honest) metrics.
    pub model: CvMetrics,
    pub prior: CvMetrics,
    /// The existing rule score used as a ranking (probabilities not meaningful, so only AUC/queue metrics).
    pub rule: CvMetrics,
    /// 95% bootstrap interval of (model AUC - rule AUC).
    pub auc_gain_vs_rule: Option<(f64, f64)>,
    pub reliability: Vec<ReliabilityBin>,
    pub gates: Vec<Gate>,
    pub passed: bool,
    /// Largest cut-off with no out-of-fold true match below half of it. `None` unless the gates passed.
    pub suggested_low_band_max_priority: Option<f64>,
    pub frozen: Vec<String>,
    /// Strongest learned coefficients (name, weight, prior weight).
    pub top_weights: Vec<(String, f64, f64)>,
    pub warnings: Vec<String>,
    pub candidate: LogisticModel,
}

fn choose_lambda(ex: &[&Example], prior: &LogisticModel, grid: &[f64], seed: u64) -> f64 {
    let default = grid.get(grid.len() / 2).copied().unwrap_or(30.0);
    if grid.len() < 2 {
        return default;
    }
    let pos = ex.iter().filter(|e| e.y).count();
    if pos < 10 || ex.len() - pos < 10 {
        // too few labels to select anything: stay near the scorecard
        return grid.iter().cloned().fold(default, f64::max);
    }
    let k = 4.min(pos);
    let folds = grouped_folds(ex, k, seed ^ 0x77);
    let mut best = (f64::INFINITY, default);
    for &lam in grid {
        let mut preds: Vec<(bool, f64)> = Vec::new();
        for f in 0..k {
            let train: Vec<&Example> = ex.iter().zip(&folds).filter(|(_, fo)| **fo != f).map(|(e, _)| *e).collect();
            let test: Vec<&Example> = ex.iter().zip(&folds).filter(|(_, fo)| **fo == f).map(|(e, _)| *e).collect();
            if train.is_empty() || test.is_empty() {
                continue;
            }
            let (th, fr) = fit_constrained(&train, prior, lam);
            let m = model_from(&th, &fr, "cv");
            preds.extend(test.iter().map(|e| (e.y, m.predict_vec(&e.x))));
        }
        let ll = log_loss(&preds);
        if ll < best.0 {
            best = (ll, lam);
        }
    }
    best.1
}

fn metrics(s: &[(bool, f64)], probabilistic: bool) -> CvMetrics {
    CvMetrics {
        auc: auc(s),
        average_precision: average_precision(s),
        brier: if probabilistic { brier(s) } else { f64::NAN },
        log_loss: if probabilistic { log_loss(s) } else { f64::NAN },
        queue_at_95_recall: queue_fraction_at_recall(s, 0.95),
        queue_at_100_recall: queue_fraction_at_recall(s, 1.0),
    }
}

/// Train, cross-validate and (maybe) pass the activation gates. Does not change any engine state.
pub fn train(examples: &[Example], opts: &TrainOptions) -> Result<TrainReport> {
    let prior = LogisticModel::prior();
    let ex: Vec<&Example> = examples.iter().collect();
    let positives = ex.iter().filter(|e| e.y).count();
    let negatives = ex.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err(Error::Invalid(format!(
            "need both confirmed matches and false positives to train (have {positives} and {negatives})"
        )));
    }
    let k = opts.folds.clamp(2, 10);
    let customers = {
        let mut g: Vec<u64> = ex.iter().map(|e| e.group).collect();
        g.sort_unstable();
        g.dedup();
        g.len()
    };
    let mut warnings = Vec::new();
    if positives < opts.gates.min_positives || negatives < opts.gates.min_negatives || ex.len() < opts.gates.min_examples {
        warnings.push(format!(
            "only {} examples ({positives} confirmed, {negatives} false positive): cross-validated figures are very noisy",
            ex.len()
        ));
    }
    if customers < k * 2 {
        warnings.push(format!("only {customers} distinct customers for {k} folds"));
    }

    // --- outer cross-validation (the whole procedure, including the ridge choice, is re-run per fold)
    let folds = grouped_folds(&ex, k, opts.seed);
    let mut oof_model: Vec<(bool, f64)> = Vec::with_capacity(ex.len());
    let mut oof_prior: Vec<(bool, f64)> = Vec::with_capacity(ex.len());
    let mut oof_rule: Vec<(bool, f64)> = Vec::with_capacity(ex.len());
    let mut triple: Vec<(bool, f64, f64)> = Vec::with_capacity(ex.len());
    for f in 0..k {
        let train: Vec<&Example> = ex.iter().zip(&folds).filter(|(_, fo)| **fo != f).map(|(e, _)| *e).collect();
        let test: Vec<&Example> = ex.iter().zip(&folds).filter(|(_, fo)| **fo == f).map(|(e, _)| *e).collect();
        if train.is_empty() || test.is_empty() || train.iter().all(|e| e.y) || train.iter().all(|e| !e.y) {
            // a fold we cannot train on: fall back to the scorecard for its test items
            for e in &test {
                let p = prior.predict_vec(&e.x);
                oof_model.push((e.y, p));
                oof_prior.push((e.y, p));
                oof_rule.push((e.y, e.rule_score));
                triple.push((e.y, p, e.rule_score));
            }
            continue;
        }
        let lam = choose_lambda(&train, &prior, &opts.lambda_grid, opts.seed.wrapping_add(f as u64));
        let (th, fr) = fit_constrained(&train, &prior, lam);
        let m = model_from(&th, &fr, "cv");
        for e in &test {
            let p = m.predict_vec(&e.x);
            oof_model.push((e.y, p));
            oof_prior.push((e.y, prior.predict_vec(&e.x)));
            oof_rule.push((e.y, e.rule_score));
            triple.push((e.y, p, e.rule_score));
        }
    }
    let model_m = metrics(&oof_model, true);
    let prior_m = metrics(&oof_prior, true);
    let rule_m = metrics(&oof_rule, false);
    let gain = bootstrap_auc_delta(&triple, opts.bootstrap_rounds, opts.seed);

    // --- final fit on everything
    let lambda = choose_lambda(&ex, &prior, &opts.lambda_grid, opts.seed);
    let (theta, frozen) = fit_constrained(&ex, &prior, lambda);
    let mut candidate = model_from(&theta, &frozen, &format!("lr-{}", opts.label));
    candidate.training = Some(TrainingSummary {
        examples: ex.len(),
        positives,
        lambda,
        cv_auc: model_m.auc,
        rule_auc: rule_m.auc,
        trained_on: opts.label.clone(),
    });
    if !candidate.frozen.is_empty() {
        warnings.push(format!(
            "coefficients frozen at zero because the data pushed them against their required sign: {}",
            candidate.frozen.join(", ")
        ));
    }

    // --- gates
    let g = &opts.gates;
    let mut gates = vec![
        Gate { name: "enough examples".into(), passed: ex.len() >= g.min_examples, detail: format!("{} (need {})", ex.len(), g.min_examples) },
        Gate { name: "enough confirmed matches".into(), passed: positives >= g.min_positives, detail: format!("{positives} (need {})", g.min_positives) },
        Gate { name: "enough false positives".into(), passed: negatives >= g.min_negatives, detail: format!("{negatives} (need {})", g.min_negatives) },
    ];
    gates.push(match gain {
        Some((lo, hi)) => Gate {
            name: "beats the rule score (95% interval of AUC gain above threshold)".into(),
            passed: lo > g.min_auc_gain_lower_bound,
            detail: format!("gain in [{lo:+.3}, {hi:+.3}], need lower bound > {:+.3}", g.min_auc_gain_lower_bound),
        },
        None => Gate { name: "beats the rule score (95% interval of AUC gain above threshold)".into(), passed: false, detail: "could not be estimated".into() },
    });
    gates.push(match (model_m.queue_at_100_recall, rule_m.queue_at_100_recall) {
        (Some(m), Some(r)) => Gate {
            name: "no workload regression to find every true match".into(),
            passed: m <= r + g.max_workload_regression,
            detail: format!("model reads {:.1}% of the queue vs rule {:.1}%", m * 100.0, r * 100.0),
        },
        _ => Gate { name: "no workload regression to find every true match".into(), passed: false, detail: "no confirmed matches".into() },
    });
    let passed = gates.iter().all(|x| x.passed);

    let suggested_low = if passed {
        let min_pos = oof_model.iter().filter(|(l, _)| *l).map(|(_, p)| *p).fold(f64::INFINITY, f64::min);
        if min_pos.is_finite() && min_pos > 0.0 {
            Some(min_pos / 2.0)
        } else {
            None
        }
    } else {
        None
    };

    let mut top: Vec<(String, f64, f64)> = (0..N_FEATURES)
        .map(|j| (FEATURE_NAMES[j].to_string(), candidate.weights[j], prior.weights[j]))
        .collect();
    top.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal));
    top.truncate(8);

    if let (Some(m), Some(r)) = (model_m.auc, rule_m.auc) {
        if m + 1e-9 < r {
            warnings.push(format!("the model is worse than the plain rule score out-of-fold (AUC {m:.3} vs {r:.3})"));
        }
    }
    Ok(TrainReport {
        examples: ex.len(),
        positives,
        negatives,
        customers,
        folds: k,
        lambda,
        model: model_m,
        prior: prior_m,
        rule: rule_m,
        auc_gain_vs_rule: gain,
        reliability: reliability(&oof_model, 10),
        gates,
        passed,
        suggested_low_band_max_priority: suggested_low,
        frozen: candidate.frozen.clone(),
        top_weights: top,
        warnings,
        candidate,
    })
}

// ----------------------------------------------------------------------------------------
// Prospective (shadow) evaluation
// ----------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShadowReport {
    pub alerts: usize,
    pub positives: usize,
    pub model_auc: Option<f64>,
    pub rule_auc: Option<f64>,
    pub model_queue_at_100_recall: Option<f64>,
    pub rule_queue_at_100_recall: Option<f64>,
    pub warnings: Vec<String>,
}

/// Compare the priority stored at alert creation with the rule score, on alerts resolved since.
/// This is the honest, prospective test: the model never saw these labels.
pub fn shadow_eval(items: &[(bool, f64, f64)]) -> ShadowReport {
    let m: Vec<(bool, f64)> = items.iter().map(|i| (i.0, i.1)).collect();
    let r: Vec<(bool, f64)> = items.iter().map(|i| (i.0, i.2)).collect();
    let mut warnings = Vec::new();
    let positives = items.iter().filter(|i| i.0).count();
    if positives < 5 {
        warnings.push(format!("only {positives} confirmed matches among resolved alerts: any comparison is anecdotal"));
    }
    ShadowReport {
        alerts: items.len(),
        positives,
        model_auc: auc(&m),
        rule_auc: auc(&r),
        model_queue_at_100_recall: queue_fraction_at_recall(&m, 1.0),
        rule_queue_at_100_recall: queue_fraction_at_recall(&r, 1.0),
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(rule: f64, dob: DobCode, id: IdCode, rare: f64) -> MatchFeatures {
        MatchFeatures {
            rule_score: rule,
            name_score: rule.min(1.0),
            strong_tokens: 2,
            subject_tokens: 2,
            entry_tokens: 2,
            dob,
            id,
            rarest_shared_idf: rare,
            mean_shared_idf: rare,
            ..Default::default()
        }
    }

    #[test]
    fn prior_is_monotone_in_the_obvious_directions() {
        let m = LogisticModel::prior();
        let base = m.predict(&feat(0.9, DobCode::Unknown, IdCode::None, 6.0));
        assert!(m.predict(&feat(0.9, DobCode::Exact, IdCode::None, 6.0)) > base);
        assert!(m.predict(&feat(0.9, DobCode::Conflict, IdCode::None, 6.0)) < base);
        assert!(m.predict(&feat(0.9, DobCode::Unknown, IdCode::Strong, 6.0)) > base);
        assert!(m.predict(&feat(0.97, DobCode::Unknown, IdCode::None, 6.0)) > base);
        // a name made of common words is less specific than one made of rare words
        assert!(m.predict(&feat(0.9, DobCode::Unknown, IdCode::None, 10.0)) > m.predict(&feat(0.9, DobCode::Unknown, IdCode::None, 2.0)));
        for p in [base, m.predict(&MatchFeatures::default())] {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn vector_is_bounded_and_named() {
        assert_eq!(FEATURE_NAMES.len(), N_FEATURES);
        let f = MatchFeatures { strong_tokens: 99, subject_tokens: 99, entry_tokens: 0, competing_hits: 10_000, rarest_shared_idf: 1e9, mean_shared_idf: -5.0, rule_score: 7.0, ..Default::default() };
        for v in f.vector() {
            assert!((0.0..=1.5).contains(&v), "{v}");
        }
    }

    #[test]
    fn explain_sums_to_the_logit() {
        let m = LogisticModel::prior();
        let f = feat(0.93, DobCode::YearOnly, IdCode::Weak, 7.0);
        let total: f64 = m.explain(&f).iter().map(|x| x.1).sum::<f64>() + m.bias;
        assert!((total - m.logit_vec(&f.vector())).abs() < 1e-9);
    }

    #[test]
    fn newton_fit_matches_a_known_solution() {
        // 1-feature-style problem: with lambda ~ 0 and well-conditioned data the fit must reach a point where the
        // gradient of the penalised likelihood vanishes.
        let mut rng = Rng(7);
        let mut ex = Vec::new();
        for i in 0..400u64 {
            let s = (rng.next() % 1000) as f64 / 1000.0;
            let f = feat(0.85 + 0.15 * s, DobCode::Unknown, IdCode::None, 5.0);
            let y = (rng.next() % 1000) as f64 / 1000.0 < s * s * 0.5;
            ex.push(Example::new(&f, y, i));
        }
        let refs: Vec<&Example> = ex.iter().collect();
        let prior = LogisticModel::prior();
        let lam = 1.0;
        let (theta, frozen) = fit_constrained(&refs, &prior, lam);
        // gradient check
        let mut theta0 = [0.0; P];
        theta0[0] = prior.bias;
        for j in 0..N_FEATURES {
            theta0[j + 1] = if frozen[j] { 0.0 } else { prior.weights[j] };
        }
        let mut lamv = [lam; P];
        lamv[0] = 1e-6;
        for a in 0..P {
            let mut g = lamv[a] * (theta[a] - theta0[a]);
            for e in &refs {
                let xa = if a == 0 { 1.0 } else if frozen[a - 1] { 0.0 } else { e.x[a - 1] };
                g += (sigmoid(zof(&theta, &e.x, &frozen)) - if e.y { 1.0 } else { 0.0 }) * xa;
            }
            assert!(g.abs() < 1e-4, "gradient {a} = {g}");
        }
    }

    #[test]
    fn strong_prior_keeps_the_scorecard_and_no_prior_follows_the_data() {
        let prior = LogisticModel::prior();
        let mut ex = Vec::new();
        let mut rng = Rng(3);
        for i in 0..300u64 {
            let rare = (rng.next() % 100) as f64 / 10.0;
            let f = feat(0.9, DobCode::Unknown, IdCode::None, rare);
            // the truth depends strongly on rarity, much more than the prior thinks
            let y = (rng.next() % 100) as f64 / 100.0 < sigmoid((rare - 8.0) * 2.0);
            ex.push(Example::new(&f, y, i));
        }
        let refs: Vec<&Example> = ex.iter().collect();
        let j = FEATURE_NAMES.iter().position(|n| *n == "rarest_shared_idf").unwrap();
        let (strong, _) = fit_constrained(&refs, &prior, 1e6);
        let (weak, _) = fit_constrained(&refs, &prior, 0.01);
        assert!((strong[j + 1] - prior.weights[j]).abs() < 0.05, "strong ridge moved the weight to {}", strong[j + 1]);
        assert!(weak[j + 1] > prior.weights[j] + 1.0, "weak ridge should learn a larger weight, got {}", weak[j + 1]);
    }

    #[test]
    fn wrong_sign_coefficients_are_frozen_not_learned() {
        // labels in which a strong identifier match is (spuriously) always a false positive
        let mut ex = Vec::new();
        for i in 0..200u64 {
            let id = if i % 2 == 0 { IdCode::Strong } else { IdCode::None };
            let f = feat(0.9, DobCode::Unknown, id, 5.0);
            ex.push(Example::new(&f, id == IdCode::None && i % 10 == 1, i));
        }
        let refs: Vec<&Example> = ex.iter().collect();
        let (theta, frozen) = fit_constrained(&refs, &LogisticModel::prior(), 0.01);
        let j = FEATURE_NAMES.iter().position(|n| *n == "id_strong").unwrap();
        assert!(frozen[j]);
        assert_eq!(theta[j + 1], 0.0);
    }

    #[test]
    fn folds_keep_customers_whole_and_spread_positives() {
        let ex: Vec<Example> = (0..100u64)
            .map(|i| Example::new(&feat(0.9, DobCode::Unknown, IdCode::None, 5.0), i % 10 == 0, i / 2))
            .collect();
        let refs: Vec<&Example> = ex.iter().collect();
        let folds = grouped_folds(&refs, 5, 1);
        let mut by_group: BTreeMap<u64, usize> = BTreeMap::new();
        for (e, f) in refs.iter().zip(&folds) {
            assert_eq!(*by_group.entry(e.group).or_insert(*f), *f, "customer split across folds");
        }
        let mut pos_per_fold = [0usize; 5];
        for (e, f) in refs.iter().zip(&folds) {
            if e.y {
                pos_per_fold[*f] += 1;
            }
        }
        assert!(pos_per_fold.iter().all(|c| *c >= 1), "{pos_per_fold:?}");
    }

    #[test]
    fn metric_reference_values() {
        // perfect separation
        let s = vec![(true, 0.9), (true, 0.8), (false, 0.2), (false, 0.1)];
        assert_eq!(auc(&s), Some(1.0));
        assert_eq!(average_precision(&s), Some(1.0));
        assert_eq!(queue_fraction_at_recall(&s, 1.0), Some(0.5));
        // a tie between a positive and a negative is resolved against us
        let t = vec![(true, 0.5), (false, 0.5)];
        assert_eq!(queue_fraction_at_recall(&t, 1.0), Some(1.0));
        // brier of a perfect coin forecast
        assert!((brier(&[(true, 0.5), (false, 0.5)]) - 0.25).abs() < 1e-12);
        assert!((log_loss(&[(true, 0.5), (false, 0.5)]) - 2f64.ln()).abs() < 1e-12);
        // single positive at the bottom: must read everything
        let w = vec![(false, 0.9), (false, 0.8), (true, 0.1)];
        assert_eq!(queue_fraction_at_recall(&w, 1.0), Some(1.0));
    }

    #[test]
    fn bands_never_demote_high_or_identifier_alerts() {
        let cfg = MlConfig { low_band_max_priority: 0.5, ..Default::default() };
        let strong_id = feat(0.9, DobCode::Unknown, IdCode::Strong, 5.0);
        assert_eq!(band(false, &strong_id, Some(0.0001), &cfg), PriorityBand::Critical);
        assert_eq!(band(true, &feat(0.9, DobCode::Unknown, IdCode::None, 5.0), Some(0.0001), &cfg), PriorityBand::Critical);
        let exact_dob = feat(0.9, DobCode::Exact, IdCode::None, 5.0);
        assert_eq!(band(false, &exact_dob, Some(0.0001), &cfg), PriorityBand::Standard);
        let plain = feat(0.9, DobCode::Unknown, IdCode::None, 5.0);
        assert_eq!(band(false, &plain, Some(0.0001), &cfg), PriorityBand::Low);
        assert_eq!(band(false, &plain, Some(0.9), &cfg), PriorityBand::Standard);
        // disabled by default
        assert_eq!(band(false, &plain, Some(0.0001), &MlConfig::default()), PriorityBand::Standard);
    }

    #[test]
    fn train_refuses_one_class_data() {
        let ex: Vec<Example> = (0..20u64).map(|i| Example::new(&feat(0.9, DobCode::Unknown, IdCode::None, 5.0), false, i)).collect();
        assert!(train(&ex, &TrainOptions::default()).is_err());
    }

    #[test]
    fn a_model_trained_on_noise_does_not_pass_the_gates() {
        // labels independent of every feature: nothing can beat the rule score, so activation must be refused
        let mut rng = Rng(99);
        let ex: Vec<Example> = (0..1500u64)
            .map(|i| {
                let f = feat(0.85 + 0.15 * (rng.next() % 1000) as f64 / 1000.0, DobCode::Unknown, IdCode::None, (rng.next() % 100) as f64 / 10.0);
                Example::new(&f, rng.next() % 100 < 6, i)
            })
            .collect();
        let r = train(&ex, &TrainOptions { bootstrap_rounds: 300, ..Default::default() }).unwrap();
        assert!(!r.passed, "noise must not pass: {:?}", r.gates);
        assert!(r.suggested_low_band_max_priority.is_none());
    }

    #[test]
    fn a_model_with_real_signal_beyond_the_rule_score_passes() {
        // truth depends on name rarity, which the rule score does not see
        let mut rng = Rng(5);
        let ex: Vec<Example> = (0..2500u64)
            .map(|i| {
                let rare = (rng.next() % 100) as f64 / 10.0;
                let f = feat(0.9 + 0.1 * (rng.next() % 100) as f64 / 100.0, DobCode::Unknown, IdCode::None, rare);
                let y = (rng.next() % 1000) as f64 / 1000.0 < sigmoid((rare - 8.0) * 1.5);
                Example::new(&f, y, i)
            })
            .collect();
        let r = train(&ex, &TrainOptions { bootstrap_rounds: 300, ..Default::default() }).unwrap();
        assert!(r.passed, "{:?}", r.gates);
        assert!(r.model.auc.unwrap() > r.rule.auc.unwrap() + 0.1);
        let low = r.suggested_low_band_max_priority.expect("cut-off suggested after passing");
        // no out-of-fold confirmed match sits below twice the cut-off
        assert!(low > 0.0 && low < 0.5);
    }

    #[test]
    fn reliability_bins_cover_everything_and_a_calibrated_model_is_calibrated() {
        let mut rng = Rng(8);
        let s: Vec<(bool, f64)> = (0..5000)
            .map(|_| {
                let p = (rng.next() % 1000) as f64 / 1000.0;
                ((rng.next() % 1000) as f64 / 1000.0 < p, p)
            })
            .collect();
        let bins = reliability(&s, 10);
        assert_eq!(bins.iter().map(|b| b.count).sum::<usize>(), 5000);
        for b in bins {
            assert!((b.mean_predicted - b.observed_rate).abs() < 0.08, "{b:?}");
        }
    }

    #[test]
    fn serialisation_roundtrip_and_fingerprint() {
        let m = LogisticModel::prior();
        let back: LogisticModel = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(m, back);
        assert_eq!(m.fingerprint(), back.fingerprint());
        let mut m2 = m.clone();
        m2.weights[0] += 0.01;
        assert_ne!(m.fingerprint(), m2.fingerprint());
        assert!(LogisticModel { weights: vec![0.0; 3], ..m.clone() }.validate().is_err());
        assert!(LogisticModel { bias: f64::NAN, ..m }.validate().is_err());
    }
}
