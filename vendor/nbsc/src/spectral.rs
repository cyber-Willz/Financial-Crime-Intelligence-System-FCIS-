//! Implements §2-6 of `ihara_zeta.rs`: the Bass-reduced quadratic
//! eigenvalue problem for the Hashimoto (non-backtracking) matrix `B`, its
//! standard linearization `M`, a matrix-free `krylov_ds::LinearOperator`
//! for `M` (so `rho_B` is estimated via Arnoldi without ever forming the
//! `2m x 2m` or `2n x 2n` matrix densely), and the sparse three-term
//! recursion that produces the `T_k` filter bank by applying it directly to
//! feature matrices rather than forming `T_k` as an `n x n` operator.

use crate::graph::Graph;
use krylov_ds::operator::LinearOperator;
use krylov_ds::{eig, Arnoldi};

/// Deterministic pseudo-random start vector (xorshift64), shared by every
/// Krylov entry point in this module so estimates stay reproducible across
/// runs without pulling in `rand` just for this.
fn deterministic_start_vector(dim: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(2685821657736338717).wrapping_add(1);
    let mut v0 = vec![0.0f64; dim];
    for vi in v0.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *vi = (state as f64 / u64::MAX as f64) - 0.5;
    }
    v0
}

/// `y = A * x` via the adjacency list — the one sparse primitive everything
/// else in this module is built from.
pub fn apply_adjacency(g: &Graph, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), g.n);
    debug_assert_eq!(y.len(), g.n);
    for v in 0..g.n {
        let mut acc = 0.0;
        for &u in &g.neighbors[v] {
            acc += x[u];
        }
        y[v] = acc;
    }
}

/// Matrix-free linear operator for the `2n x 2n` linearization
///
/// ```text
/// M = [ A     I-D ]
///     [ I_n   0   ]
/// ```
///
/// from §4 of `ihara_zeta.rs`. `M`'s eigenvalues are exactly the roots `mu`
/// of the quadratic eigenvalue problem (★), i.e. the non-trivial part of
/// the Hashimoto spectrum (all Hashimoto eigenvalues except the `2(m-n)`
/// trivial ones sitting at +-1). Never materializes `A`, `D`, or `M`: each
/// `apply` is one sparse adjacency mat-vec plus two elementwise passes.
pub struct HashimotoLinearization<'a> {
    pub graph: &'a Graph,
    pub degrees: Vec<f64>,
}

impl<'a> HashimotoLinearization<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self { graph, degrees: graph.degrees() }
    }
}

impl<'a> LinearOperator<f64> for HashimotoLinearization<'a> {
    fn dim(&self) -> usize {
        2 * self.graph.n
    }

    fn apply(&self, v: &[f64], out: &mut [f64]) {
        let n = self.graph.n;
        let (x, y) = v.split_at(n);
        let (out_top, out_bottom) = out.split_at_mut(n);

        // out_top = A*x + (I - D)*y
        apply_adjacency(self.graph, x, out_top);
        for i in 0..n {
            out_top[i] += (1.0 - self.degrees[i]) * y[i];
        }
        // out_bottom = x
        out_bottom.copy_from_slice(x);
    }
}

/// Estimate `rho_B`, the Perron-Frobenius spectral radius of the Hashimoto
/// matrix (§6): run Arnoldi on the matrix-free linearization and take the
/// largest-modulus Ritz value. By Perron-Frobenius this radius is attained
/// by a real, non-negative eigenvalue, so `rho_B` itself is always real
/// even though most of the surrounding spectrum is complex.
///
/// `krylov_dim` controls the Krylov subspace size; `2 * n.min(40)` is a
/// reasonable default that converges the extremal eigenvalue reliably
/// without approaching `O(n^2)` memory for the dense Hessenberg block.
pub fn estimate_spectral_radius(graph: &Graph, krylov_dim: usize, seed: u64) -> f64 {
    let op = HashimotoLinearization::new(graph);
    let dim = op.dim();
    let m = krylov_dim.min(dim).max(1);
    let v0 = deterministic_start_vector(dim, seed);

    let arnoldi = Arnoldi::new(m, 1e-12);
    let result = arnoldi.run(&op, &v0).expect("Arnoldi failed on Hashimoto linearization");
    let ritz = eig::arnoldi_ritz_values(&result);
    ritz.iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).fold(0.0, f64::max)
}

// =============================================================================
// Exponential tilting of the non-backtracking spectrum: rho_B(theta), drho/dtheta
// =============================================================================
//
// Exponential tilting reweights a distribution/operator by `exp(theta * f)`
// for some observable `f` and parameter `theta`, then renormalizes — the
// same construction whether `f` is a random-variable outcome (large
// deviations / Cramer's theorem), a Hamiltonian term (statistical physics
// Gibbs measures), or a Hermitian generator (quantum state reweighting).
// Applied to a nonnegative matrix `B`, the tilt is `B(theta)_ij = B_ij *
// exp(theta * f_j)`: the Perron-Frobenius eigenvalue `rho(theta)` of the
// tilted matrix is then the scaled cumulant generating function (up to a
// log) of the additive functional `sum_k f(step_k)` along a long random
// walk governed by `B`, and `d(log rho)/dtheta` at `theta` is the mean of
// `f` under the exponentially-tilted walk (this is exactly Varadhan's
// lemma / the Perron-root perturbation identity used below).
//
// Applied here to the Hashimoto (non-backtracking) matrix `B`, tilting by a
// per-arc observable lets `rho_B(theta)` answer questions the *untilted*
// radius cannot: e.g. "how much does the non-backtracking spectral radius
// grow if I bias walks toward high-risk vertices?" — `drho/dtheta` at
// `theta=0` is exactly that sensitivity, a first-order gradient with
// respect to the bias strength, useful as a compliance/anomaly-scoring
// primitive without ever forming or solving for `B(theta)` at a new
// `theta` from scratch.
//
// The `2n x 2n` Bass-reduced linearization `M` above is a trick specific to
// the *untilted* `B` (its derivation leans on `B` being 0/1-valued); a
// non-uniform tilt breaks that identity, so this section works directly on
// the natural `2m`-dimensional directed-arc ("darc") space `B` lives on.

/// Directed-arc index: every undirected edge `(u, v)` yields two directed
/// arcs `u -> v` and `v -> u`, each with a stable integer id. This is the
/// domain `B` (and any exponential tilt of it) is naturally indexed by,
/// distinct from the `2n`-dimensional space of [`HashimotoLinearization`].
/// Arc `2k` and `2k+1` are always the two directions of edge `k`.
pub struct DarcIndex {
    /// `arcs[i] = (u, v)` for the `i`-th directed arc.
    pub arcs: Vec<(usize, usize)>,
    /// `out_arcs[v]` = ids of arcs leaving `v`.
    out_arcs: Vec<Vec<usize>>,
    /// `in_arcs[v]` = ids of arcs arriving at `v`.
    in_arcs: Vec<Vec<usize>>,
    /// `reverse_id[i]` = id of the arc `(v, u)` given `arcs[i] == (u, v)`.
    reverse_id: Vec<usize>,
}

impl DarcIndex {
    pub fn build(graph: &Graph) -> Self {
        let m = graph.m();
        let mut arcs = Vec::with_capacity(2 * m);
        let mut out_arcs = vec![Vec::new(); graph.n];
        let mut in_arcs = vec![Vec::new(); graph.n];
        let mut reverse_id = Vec::with_capacity(2 * m);
        for &(u, v) in &graph.edges {
            let id_uv = arcs.len();
            arcs.push((u, v));
            let id_vu = arcs.len();
            arcs.push((v, u));
            out_arcs[u].push(id_uv);
            in_arcs[v].push(id_uv);
            out_arcs[v].push(id_vu);
            in_arcs[u].push(id_vu);
            reverse_id.push(id_vu); // reverse of id_uv
            reverse_id.push(id_uv); // reverse of id_vu
        }
        Self { arcs, out_arcs, in_arcs, reverse_id }
    }

    /// Dimension of the darc space, `2m`.
    pub fn dim(&self) -> usize {
        self.arcs.len()
    }
}

/// A per-arc observable `f: arc -> R` defining an exponential tilt
/// `B(theta)_ij = B_ij * exp(theta * f_j)` of the non-backtracking matrix
/// (the weight is attached to the *destination* arc `j`, i.e. to the arc a
/// step lands on — the natural convention for tilting an additive
/// functional accumulated one step at a time).
#[derive(Debug, Clone)]
pub struct ArcWeights {
    pub values: Vec<f64>,
}

impl ArcWeights {
    /// The trivial uniform tilt `f = 1` on every arc. Because `B(theta) =
    /// e^theta * B` exactly in this case, `rho_B(theta) = e^theta *
    /// rho_B(0)` and `drho/dtheta = rho_B(theta)` in closed form — this is
    /// the sanity check `tilted_radius_matches_uniform_closed_form` below
    /// exercises against the general (non-closed-form) machinery.
    pub fn uniform(darcs: &DarcIndex) -> Self {
        Self { values: vec![1.0; darcs.dim()] }
    }

    /// Tilt by a per-vertex potential `phi`, evaluated at the arc's head
    /// (the vertex a step arrives at): `f(u -> v) = phi[v]`. The common
    /// case — bias non-backtracking walks toward (`theta > 0`) or away
    /// from (`theta < 0`) vertices scoring high on some feature (a risk
    /// score, a degree, an anomaly flag, ...).
    pub fn from_head_potential(darcs: &DarcIndex, phi: &[f64]) -> Self {
        let values = darcs.arcs.iter().map(|&(_, v)| phi[v]).collect();
        Self { values }
    }
}

/// Matrix-free forward tilted operator `B(theta)` on the `2m`-dimensional
/// darc space: `(B(theta) x)_i = sum_{j successor of i} exp(theta * f_j) *
/// x_j`, where "successor of `i = (u,v)`" means every arc `(v, w)` with `w
/// != u` (the non-backtracking constraint — excludes only the reverse arc).
struct TiltedForward<'a> {
    darcs: &'a DarcIndex,
    weights: &'a ArcWeights,
    theta: f64,
}

impl<'a> LinearOperator<f64> for TiltedForward<'a> {
    fn dim(&self) -> usize {
        self.darcs.dim()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.darcs.dim() {
            let (_, v) = self.darcs.arcs[i];
            let rev = self.darcs.reverse_id[i];
            let mut acc = 0.0;
            for &j in &self.darcs.out_arcs[v] {
                if j == rev {
                    continue; // non-backtracking: can't step straight back
                }
                acc += (self.theta * self.weights.values[j]).exp() * x[j];
            }
            y[i] = acc;
        }
    }
}

/// Matrix-free transpose `B(theta)^T`. Since `B(theta)_pq = B_pq *
/// exp(theta * f_q)`, transposing gives `(B(theta)^T)_{q,p} = B_pq *
/// exp(theta * f_q)`: the tilt weight is a function of the *row* index `q`
/// alone here, so it factors out of the inner sum entirely.
struct TiltedTranspose<'a> {
    darcs: &'a DarcIndex,
    weights: &'a ArcWeights,
    theta: f64,
}

impl<'a> LinearOperator<f64> for TiltedTranspose<'a> {
    fn dim(&self) -> usize {
        self.darcs.dim()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.darcs.dim() {
            let (u, _) = self.darcs.arcs[i];
            let rev = self.darcs.reverse_id[i];
            let mut acc = 0.0;
            for &p in &self.darcs.in_arcs[u] {
                if p == rev {
                    continue;
                }
                acc += x[p];
            }
            y[i] = acc * (self.theta * self.weights.values[i]).exp();
        }
    }
}

/// `(dB/dtheta * x)_i = sum_{j successor of i} f_j * exp(theta * f_j) *
/// x_j` — the same sparsity pattern as [`TiltedForward::apply`], with each
/// surviving term additionally scaled by the tilt weight of the column it
/// came from. This is `d/dtheta (B(theta) x)` at fixed `x`, which is what
/// the eigenvalue-derivative formula in [`tilted_spectral_radius`] needs.
fn apply_forward_dtheta(darcs: &DarcIndex, weights: &ArcWeights, theta: f64, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; darcs.dim()];
    for i in 0..darcs.dim() {
        let (_, v) = darcs.arcs[i];
        let rev = darcs.reverse_id[i];
        let mut acc = 0.0;
        for &j in &darcs.out_arcs[v] {
            if j == rev {
                continue;
            }
            let fj = weights.values[j];
            acc += fj * (theta * fj).exp() * x[j];
        }
        y[i] = acc;
    }
    y
}

/// The dominant real Ritz pair by magnitude, i.e. the Perron pair for a
/// nonnegative, irreducible operator (both [`TiltedForward`] and
/// [`TiltedTranspose`] have all-nonnegative entries whenever `theta *
/// weights` doesn't overflow, since `exp(...) > 0` always).
fn dominant_real_pair(pairs: &[eig::RitzPair<f64>]) -> (f64, Vec<f64>) {
    let best = pairs
        .iter()
        .max_by(|a, b| a.value.abs().partial_cmp(&b.value.abs()).unwrap())
        .expect(
            "no real Ritz pair found for the tilted Hashimoto operator; \
             try a larger krylov_dim (the Perron root should always be real)",
        );
    (best.value, best.vector.clone())
}

/// Result of [`tilted_spectral_radius`]: the tilted Perron root and its
/// first derivative in `theta`, at a single `theta`.
#[derive(Debug, Clone, Copy)]
pub struct TiltedSpectralRadius {
    pub theta: f64,
    /// `rho_B(theta)`, the Perron-Frobenius eigenvalue of `B(theta)`.
    pub rho: f64,
    /// `drho/dtheta` at this `theta`, via the standard non-symmetric
    /// eigenvalue perturbation identity `<v, dB/dtheta w> / <v, w>` for
    /// right/left Perron eigenvectors `w`, `v` (Hellmann-Feynman for a
    /// general — not necessarily symmetric — matrix). Equivalently,
    /// `d(log rho)/dtheta` is the mean of `f` under the walk measure
    /// exponentially tilted by `theta` (Varadhan's lemma).
    pub drho_dtheta: f64,
}

/// General `nbsc::spectral` primitive: the exponentially-tilted
/// non-backtracking spectral radius `rho_B(theta)` and its derivative
/// `drho/dtheta`, for an arbitrary per-arc observable `weights`.
///
/// Computes `rho(theta)` via Arnoldi on the matrix-free forward tilted
/// operator (right Perron vector `w`), separately via Arnoldi on the
/// transpose (left Perron vector `v`), then closes `drho/dtheta` in
/// closed form from `w` and `v` — no repeated re-solving at nearby
/// `theta` values (contrast with a finite-difference estimate, which the
/// `theta=0`, uniform-weight test below cross-checks this against).
///
/// `theta = 0` recovers the plain (untilted) Hashimoto spectral radius,
/// i.e. `tilted_spectral_radius(..., 0.0, ...).rho` should agree with
/// [`estimate_spectral_radius`] up to Krylov truncation error.
pub fn tilted_spectral_radius(
    darcs: &DarcIndex,
    weights: &ArcWeights,
    theta: f64,
    krylov_dim: usize,
    seed: u64,
) -> TiltedSpectralRadius {
    assert_eq!(
        darcs.dim(),
        weights.values.len(),
        "ArcWeights must have one entry per darc"
    );
    let dim = darcs.dim();
    let m = krylov_dim.min(dim).max(1);
    let v0 = deterministic_start_vector(dim, seed);
    let arnoldi = Arnoldi::new(m, 1e-12);

    let fwd = TiltedForward { darcs, weights, theta };
    let fwd_result = arnoldi.run(&fwd, &v0).expect("Arnoldi failed on tilted Hashimoto (forward)");
    let (rho, w) = dominant_real_pair(&eig::arnoldi_real_ritz_pairs(&fwd_result));

    let bwd = TiltedTranspose { darcs, weights, theta };
    let bwd_result = arnoldi.run(&bwd, &v0).expect("Arnoldi failed on tilted Hashimoto (transpose)");
    let (_rho_t, v) = dominant_real_pair(&eig::arnoldi_real_ritz_pairs(&bwd_result));

    let dtheta_w = apply_forward_dtheta(darcs, weights, theta, &w);
    let numer: f64 = v.iter().zip(dtheta_w.iter()).map(|(a, b)| a * b).sum();
    let denom: f64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();

    TiltedSpectralRadius { theta, rho, drho_dtheta: numer / denom }
}

// =============================================================================
// Legendre transform of the scaled CGF: the large-deviations rate function
// I(x) = sup_theta ( theta * x - Lambda(theta) ), Lambda(theta) = log rho_B(theta)
// =============================================================================
//
// Once `Lambda(theta) = log rho_B(theta)` is known to be finite and convex
// near the origin -- exactly what `tilt_check`'s monotone-`d(log rho)/dtheta`
// assertion verifies empirically -- the Gartner-Ellis theorem identifies the
// large-deviations rate function governing the empirical mean of the tilt
// observable `f` along a long non-backtracking walk as the Legendre-Fenchel
// transform of `Lambda`:
//
//     I(x) = sup_theta ( theta * x - Lambda(theta) )
//
// Because `Lambda` is convex and smooth here, the sup is attained at the
// unique `theta*` solving the first-order condition `Lambda'(theta*) = x`,
// and `I(x) = theta* * x - Lambda(theta*)` there. `Lambda'(theta) =
// drho_dtheta / rho` is already available in closed form from
// `tilted_spectral_radius`, so `theta*` is found by bisecting `Lambda'(theta)
// - x` (monotone non-decreasing in `theta` by convexity of `Lambda`) rather
// than by a general-purpose optimizer.

/// `(Lambda(theta), Lambda'(theta))` at a single `theta`, i.e. `(log
/// rho_B(theta), drho_dtheta / rho)`. Bundled together since every
/// Legendre-transform step needs both.
fn scaled_cgf(
    darcs: &DarcIndex,
    weights: &ArcWeights,
    theta: f64,
    krylov_dim: usize,
    seed: u64,
) -> (f64, f64) {
    let t = tilted_spectral_radius(darcs, weights, theta, krylov_dim, seed);
    (t.rho.ln(), t.drho_dtheta / t.rho)
}

/// Result of [`legendre_rate`] at a single point `x`: the large-deviations
/// rate `I(x)`, the maximizing (equivalently, "typical-value-inducing")
/// tilt `theta_star`, and `Lambda(theta_star)` for diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct RateFunctionPoint {
    pub x: f64,
    pub theta_star: f64,
    pub lambda_theta_star: f64,
    /// `I(x) = theta_star * x - lambda_theta_star`.
    pub rate: f64,
}

/// Legendre-Fenchel transform of `Lambda(theta) = log rho_B(theta)` at a
/// single point `x`: the large-deviations rate function `I(x) = sup_theta
/// (theta * x - Lambda(theta))` for the empirical mean of the tilt
/// observable `weights` along a long non-backtracking walk on `darcs`.
///
/// Solves `Lambda'(theta*) = x` by bisection, expanding a bracket outward
/// from `theta = 0` (doubling each side) until it contains a sign change of
/// `Lambda'(theta) - x`, then bisecting down to `1e-8` in `theta`. Valid
/// because `Lambda` is convex, so `Lambda'` is monotone non-decreasing --
/// the same convexity `tilt_check` verifies empirically for `log rho_B`.
///
/// # Panics
/// Panics if no sign change is found within `theta in [-2^60, 2^60]`. That
/// means `x` lies outside `Lambda`'s attainable derivative range (its
/// domain's boundary slopes), where the true rate is `I(x) = +infinity`; no
/// finite `theta*` can represent that, so this returns an error condition
/// (panic) rather than a silently-truncated finite answer.
pub fn legendre_rate(
    darcs: &DarcIndex,
    weights: &ArcWeights,
    x: f64,
    krylov_dim: usize,
    seed: u64,
) -> RateFunctionPoint {
    const TOL: f64 = 1e-8;
    const MAX_EXPANSIONS: u32 = 60;
    const MAX_BISECTIONS: u32 = 200;

    let deriv = |theta: f64| -> f64 { scaled_cgf(darcs, weights, theta, krylov_dim, seed).1 };
    let g = |theta: f64| -> f64 { deriv(theta) - x };

    // g is monotone non-decreasing in theta (convexity of Lambda). Expand a
    // bracket [lo, hi] around 0 until it contains a sign change -- but check
    // each endpoint against TOL as soon as it's computed and stop there:
    // floating-point noise can leave a converged endpoint just barely on
    // the "wrong" side of zero (e.g. -1e-14 for a true derivative of 0),
    // and treating that as "needs more expansion" would walk `theta`
    // outward into a regime where the tilted matrix is badly scaled and
    // Arnoldi may fail to resolve a real Perron root at all -- a needless
    // robustness cost when the root was already found.
    let mut lo = -1.0_f64;
    let mut hi = 1.0_f64;
    let mut g_lo = g(lo);
    if g_lo.abs() < TOL {
        let (lambda_theta_star, _) = scaled_cgf(darcs, weights, lo, krylov_dim, seed);
        return RateFunctionPoint { x, theta_star: lo, lambda_theta_star, rate: lo * x - lambda_theta_star };
    }
    let mut g_hi = g(hi);
    if g_hi.abs() < TOL {
        let (lambda_theta_star, _) = scaled_cgf(darcs, weights, hi, krylov_dim, seed);
        return RateFunctionPoint { x, theta_star: hi, lambda_theta_star, rate: hi * x - lambda_theta_star };
    }
    let mut expansions = 0;
    while g_lo > 0.0 || g_hi < 0.0 {
        expansions += 1;
        assert!(
            expansions <= MAX_EXPANSIONS,
            "legendre_rate: could not bracket theta* for x={x} within \
             theta in [-2^{MAX_EXPANSIONS}, 2^{MAX_EXPANSIONS}]; x is \
             likely outside Lambda's attainable derivative range \
             (I(x) = +infinity there)"
        );
        if g_lo > 0.0 {
            lo *= 2.0;
            g_lo = g(lo);
            if g_lo.abs() < TOL {
                let (lambda_theta_star, _) = scaled_cgf(darcs, weights, lo, krylov_dim, seed);
                return RateFunctionPoint { x, theta_star: lo, lambda_theta_star, rate: lo * x - lambda_theta_star };
            }
        }
        if g_hi < 0.0 {
            hi *= 2.0;
            g_hi = g(hi);
            if g_hi.abs() < TOL {
                let (lambda_theta_star, _) = scaled_cgf(darcs, weights, hi, krylov_dim, seed);
                return RateFunctionPoint { x, theta_star: hi, lambda_theta_star, rate: hi * x - lambda_theta_star };
            }
        }
    }

    let theta_star = {
        let mut theta_star = 0.5 * (lo + hi);
        for _ in 0..MAX_BISECTIONS {
            let g_mid = g(theta_star);
            if g_mid.abs() < TOL || (hi - lo) < TOL {
                break;
            }
            if g_mid > 0.0 {
                hi = theta_star;
            } else {
                lo = theta_star;
            }
            theta_star = 0.5 * (lo + hi);
        }
        theta_star
    };

    let (lambda_theta_star, _) = scaled_cgf(darcs, weights, theta_star, krylov_dim, seed);
    RateFunctionPoint {
        x,
        theta_star,
        lambda_theta_star,
        rate: theta_star * x - lambda_theta_star,
    }
}

/// Evaluate [`legendre_rate`] at each `x` in `xs`. Each point's bisection is
/// bracketed independently from `theta = 0` (deliberately not warm-started
/// from the previous point's `theta_star`), so the resulting curve does not
/// depend on the order of `xs`.
pub fn legendre_rate_curve(
    darcs: &DarcIndex,
    weights: &ArcWeights,
    xs: &[f64],
    krylov_dim: usize,
    seed: u64,
) -> Vec<RateFunctionPoint> {
    xs.iter().map(|&x| legendre_rate(darcs, weights, x, krylov_dim, seed)).collect()
}

/// Matrix-free `LinearOperator` for the plain adjacency matrix `A` —
/// distinct from [`HashimotoLinearization`], which linearizes the
/// non-backtracking matrix `B`. `A` is symmetric (undirected graph), so
/// this is a valid input to `krylov_ds::Lanczos` (unlike `M`, which is not
/// symmetric and requires the general `Arnoldi`).
struct AdjacencyOperator<'a> {
    graph: &'a Graph,
}

impl<'a> LinearOperator<f64> for AdjacencyOperator<'a> {
    fn dim(&self) -> usize {
        self.graph.n
    }
    fn apply(&self, v: &[f64], out: &mut [f64]) {
        apply_adjacency(self.graph, v, out);
    }
}

/// Diagnostic: the operator (spectral) norm of the plain adjacency matrix,
/// `||A||_2 = max(|lambda_min(A)|, |lambda_max(A)|)`. Since `A` is real
/// symmetric, this is exactly its largest-magnitude eigenvalue, found via
/// Lanczos (which — unlike Arnoldi on the non-symmetric Hashimoto
/// linearization `M` — gives real, well-conditioned Ritz values here).
///
/// Exists to test a specific hypothesis raised by an empirical anomaly: if
/// `adjacency_operator_norm(g) / rho_b > 1`, the rescaled adjacency tap
/// `A / rho_B` used by [`NbscFilterBank`] is an *expansive* map (operator
/// norm > 1), unlike GCN's symmetric-normalized `D^-1/2 (A+I) D^-1/2`,
/// which is non-expansive by construction (its norm is exactly 1). An
/// expansive propagation step compounds across stacked layers, which would
/// explain both growing Dirichlet energy with depth and the increasing
/// cross-seed variance observed empirically at depth 3 on Cora — ordinary
/// over-smoothing predicts *shrinking* energy, not growing energy with
/// widening spread. This function does not draw that conclusion itself;
/// it just gives you the number needed to check it.
pub fn adjacency_operator_norm(graph: &Graph, krylov_dim: usize, seed: u64) -> f64 {
    let op = AdjacencyOperator { graph };
    let n = op.dim();
    let m = krylov_dim.min(n).max(1);
    let v0 = deterministic_start_vector(n, seed);

    let lanczos = krylov_ds::Lanczos::new(m, 1e-12, krylov_ds::Reorthogonalization::Full);
    let result = lanczos.run(&op, &v0).expect("Lanczos failed on adjacency operator");
    let pairs = krylov_ds::eig::lanczos_ritz_pairs(&result);
    pairs.iter().map(|p| p.value.abs()).fold(0.0, f64::max)
}

/// Direct dense construction of `B` (`2m x 2m`) for small graphs only —
/// used strictly as a ground-truth oracle in tests to verify both the Bass
/// reduction identity and the matrix-free `M` operator against a brute-force
/// eigendecomposition. `O(m^2)` memory; never call this outside tests or on
/// graphs beyond a few hundred edges.
#[cfg(test)]
pub(crate) fn dense_hashimoto_matrix(graph: &Graph) -> (Vec<(usize, usize)>, nalgebra::DMatrix<f64>) {
    // Directed edges: each undirected edge (u,v) becomes (u->v) and (v->u).
    let mut darcs: Vec<(usize, usize)> = Vec::with_capacity(2 * graph.m());
    for &(u, v) in &graph.edges {
        darcs.push((u, v));
        darcs.push((v, u));
    }
    let d2m = darcs.len();
    let mut b = nalgebra::DMatrix::<f64>::zeros(d2m, d2m);
    for (i, &(_u, v)) in darcs.iter().enumerate() {
        for (j, &(vp, w)) in darcs.iter().enumerate() {
            if v == vp && w != darcs[i].0 {
                b[(i, j)] = 1.0;
            }
        }
    }
    (darcs, b)
}

/// A dense node-feature matrix, `n` rows x `f` columns, row-major.
#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    pub n: usize,
    pub f: usize,
    pub data: Vec<f64>,
}

impl FeatureMatrix {
    pub fn zeros(n: usize, f: usize) -> Self {
        Self { n, f, data: vec![0.0; n * f] }
    }

    pub fn from_rows(n: usize, f: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), n * f);
        Self { n, f, data }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.f..(i + 1) * self.f]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.f..(i + 1) * self.f]
    }
}

/// Applies `A` and `(D-I)` to every column of a feature matrix at once
/// (i.e. `Y = A @ X`, `Y = (D-I) @ X`), which is what the three-term
/// recursion actually needs — this is the sparse `O(|E| * f)`-per-tap
/// implementation of §5's recursion, applied directly to features rather
/// than materializing `T_k` as an `n x n` operator (an explicit improvement
/// over the "TODO: replace dense T_k with a sparse kernel" note in §8 of
/// `ihara_zeta.rs` — there is simply no need to ever form `T_k` at all when
/// all that's wanted is `T_k @ X`).
fn adjacency_matmul(graph: &Graph, x: &FeatureMatrix, out: &mut FeatureMatrix) {
    debug_assert_eq!(x.n, graph.n);
    debug_assert_eq!(out.n, graph.n);
    debug_assert_eq!(out.f, x.f);
    for v in 0..graph.n {
        let row_out = out.row_mut(v);
        row_out.fill(0.0);
        for &u in &graph.neighbors[v] {
            let row_u = x.row(u);
            for (o, &xu) in row_out.iter_mut().zip(row_u.iter()) {
                *o += xu;
            }
        }
    }
}

fn diag_dm1_matmul(degrees: &[f64], x: &FeatureMatrix, out: &mut FeatureMatrix) {
    for v in 0..x.n {
        let scale = degrees[v] - 1.0;
        let row_x = x.row(v);
        let row_out = out.row_mut(v);
        for (o, &xv) in row_out.iter_mut().zip(row_x.iter()) {
            *o = scale * xv;
        }
    }
}

/// A precomputed, graph-fixed (X-independent) filter bank: holds `rho_B`
/// and the graph itself, and applies the rescaled three-term recursion of
/// §6:
///
/// ```text
/// T_0 = I
/// T_1 = A / rho_B
/// T_{k+1} = (2A/rho_B) T_k - ((D-I)/rho_B^2) T_{k-1}
/// ```
///
/// directly to a feature matrix, sparse, at `O(K * |E| * f)` total cost —
/// exactly the same complexity class as ChebNet/GCN per §5.
pub struct NbscFilterBank {
    pub rho_b: f64,
    degrees: Vec<f64>,
}

impl NbscFilterBank {
    pub fn build(graph: &Graph, krylov_dim: usize, seed: u64) -> Self {
        let rho_b = estimate_spectral_radius(graph, krylov_dim, seed).max(1e-6);
        Self { rho_b, degrees: graph.degrees() }
    }

    /// Returns `[T_0 X, T_1 X, ..., T_K X]`.
    pub fn apply_taps(&self, graph: &Graph, x: &FeatureMatrix, k: usize) -> Vec<FeatureMatrix> {
        assert_eq!(x.n, graph.n);
        let mut taps = Vec::with_capacity(k + 1);
        taps.push(x.clone()); // T_0 X = X

        if k == 0 {
            return taps;
        }

        // T_1 X = (A / rho_B) X
        let mut t1 = FeatureMatrix::zeros(x.n, x.f);
        adjacency_matmul(graph, x, &mut t1);
        for v in t1.data.iter_mut() {
            *v /= self.rho_b;
        }
        taps.push(t1);

        for kk in 1..k {
            let prev = &taps[kk]; // T_k X
            let prev2 = &taps[kk - 1]; // T_{k-1} X

            let mut a_prev = FeatureMatrix::zeros(x.n, x.f);
            adjacency_matmul(graph, prev, &mut a_prev);

            let mut d_prev2 = FeatureMatrix::zeros(x.n, x.f);
            diag_dm1_matmul(&self.degrees, prev2, &mut d_prev2);

            let mut next = FeatureMatrix::zeros(x.n, x.f);
            let rho2 = self.rho_b * self.rho_b;
            for i in 0..next.data.len() {
                next.data[i] = (2.0 / self.rho_b) * a_prev.data[i] - (1.0 / rho2) * d_prev2.data[i];
            }
            taps.push(next);
        }
        taps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{random_near_regular, random_tree, stochastic_block_model};
    use approx::assert_relative_eq;

    /// §3: Bass's determinant identity. For a small graph, verify
    /// `det(I - u*B) == (1-u^2)^(m-n) * det(I - u*A + u^2*(D-I))`
    /// at several sample points `u`, confirming the reduction the whole
    /// pipeline (and the matrix-free `M` operator) rests on.
    #[test]
    fn bass_reduction_identity_holds() {
        let (g, _) = stochastic_block_model(3, 5, 0.6, 0.1, 11);
        assert!(g.is_connected());
        let (darcs, b) = dense_hashimoto_matrix(&g);
        let d2m = darcs.len();
        let n = g.n;
        let m = g.m();

        let a = nalgebra::DMatrix::from_fn(n, n, |i, j| {
            if g.neighbors[i].contains(&j) {
                1.0
            } else {
                0.0
            }
        });
        let degrees = g.degrees();
        let d_minus_i = nalgebra::DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                degrees[i] - 1.0
            } else {
                0.0
            }
        });

        for &u in &[0.05, 0.1, 0.15, -0.1] {
            let lhs = (nalgebra::DMatrix::<f64>::identity(d2m, d2m) - b.clone() * u).determinant();
            let rhs_poly =
                (nalgebra::DMatrix::<f64>::identity(n, n) - a.clone() * u + d_minus_i.clone() * u * u)
                    .determinant();
            let rhs = (1.0 - u * u).powi(m as i32 - n as i32) * rhs_poly;
            assert_relative_eq!(lhs, rhs, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    /// `adjacency_operator_norm` (Lanczos on the matrix-free adjacency
    /// operator) should agree with a brute-force dense eigendecomposition
    /// of `A`, and — since `A` is symmetric — should also agree with the
    /// symmetric-normalized GCN propagator's known operator norm ceiling of
    /// 1 being violated or not, which is the whole point of this function.
    #[test]
    fn adjacency_operator_norm_matches_dense_ground_truth() {
        let g = random_near_regular(30, 4, 7);
        let n = g.n;
        let a = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| {
            if g.neighbors[i].contains(&j) { 1.0 } else { 0.0 }
        });
        let dense_norm = a
            .symmetric_eigenvalues()
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);

        let krylov_norm = adjacency_operator_norm(&g, 25, 5);
        assert_relative_eq!(krylov_norm, dense_norm, epsilon = 1e-6, max_relative = 1e-4);
    }

    /// The matrix-free linearization `M`'s eigenvalues (via a *dense*
    /// eigendecomposition here, not Arnoldi, to isolate correctness of the
    /// operator from Krylov convergence) must reproduce the roots of (★)
    /// obtained directly from the quadratic eigenvalue problem, and hence
    /// (via Bass) the non-trivial part of the true Hashimoto spectrum.
    #[test]
    fn linearization_matches_dense_hashimoto_nontrivial_spectrum() {
        let g = random_near_regular(24, 3, 5);
        assert!(g.is_connected());
        let (_darcs, b) = dense_hashimoto_matrix(&g);

        let mut b_eig: Vec<f64> = b
            .complex_eigenvalues()
            .iter()
            .filter(|z| z.im.abs() < 1e-6)
            .map(|z| z.re)
            .filter(|&re| (re.abs() - 1.0).abs() > 1e-3) // drop the trivial +-1 eigenvalues (§2)
            .collect();
        b_eig.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = g.n;
        let op = HashimotoLinearization::new(&g);
        let m_dense = nalgebra::DMatrix::from_fn(2 * n, 2 * n, |i, j| {
            let mut e = vec![0.0; 2 * n];
            e[j] = 1.0;
            let mut out = vec![0.0; 2 * n];
            op.apply(&e, &mut out);
            out[i]
        });
        let mut m_eig: Vec<f64> = m_dense
            .complex_eigenvalues()
            .iter()
            .filter(|z| z.im.abs() < 1e-6)
            .map(|z| z.re)
            .collect();
        m_eig.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Every real eigenvalue of B (away from +-1) should also appear in
        // M's spectrum (M additionally carries the trivial +-1 pairs, so we
        // check containment, not equality of the full multisets).
        for be in &b_eig {
            let found = m_eig.iter().any(|me| (me - be).abs() < 1e-3);
            assert!(found, "Hashimoto eigenvalue {be} not found in linearization spectrum");
        }
    }

    /// `rho_B` from Krylov (matrix-free) Arnoldi should agree with the
    /// dense ground truth to a small tolerance.
    #[test]
    fn spectral_radius_matches_dense_ground_truth() {
        let g = random_near_regular(30, 4, 9);
        assert!(g.is_connected());
        let (_darcs, b) = dense_hashimoto_matrix(&g);
        let dense_rho = b.complex_eigenvalues().iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).fold(0.0, f64::max);

        let krylov_rho = estimate_spectral_radius(&g, 40, 3);
        assert_relative_eq!(krylov_rho, dense_rho, epsilon = 1e-4, max_relative = 1e-3);
    }

    /// Sanity/regression check for the sparse three-term feature recursion:
    /// compare `T_2 X` from `NbscFilterBank::apply_taps` against a direct
    /// dense matrix computation of `T_2 = (2A/rho)*T_1 - ((D-I)/rho^2)*T_0`.
    #[test]
    fn filter_bank_recursion_matches_dense_reference() {
        let g = random_near_regular(20, 3, 21);
        let bank = NbscFilterBank::build(&g, 30, 1);
        let f = 3;
        let mut x = FeatureMatrix::zeros(g.n, f);
        let mut seed = 12345u64;
        for v in x.data.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *v = (seed as f64 / u64::MAX as f64) - 0.5;
        }

        let taps = bank.apply_taps(&g, &x, 2);
        assert_eq!(taps.len(), 3);

        let a = nalgebra::DMatrix::from_fn(g.n, g.n, |i, j| {
            if g.neighbors[i].contains(&j) {
                1.0
            } else {
                0.0
            }
        });
        let degrees = g.degrees();
        let d_minus_i = nalgebra::DMatrix::from_fn(g.n, g.n, |i, j| {
            if i == j {
                degrees[i] - 1.0
            } else {
                0.0
            }
        });
        let x_dense = nalgebra::DMatrix::from_row_slice(g.n, f, &x.data);
        let t0 = x_dense.clone();
        let t1 = (&a * &x_dense) / bank.rho_b;
        let t2 = (&a * &t1) * (2.0 / bank.rho_b) - (&d_minus_i * &t0) * (1.0 / (bank.rho_b * bank.rho_b));

        for i in 0..g.n {
            for j in 0..f {
                assert_relative_eq!(taps[2].row(i)[j], t2[(i, j)], epsilon = 1e-8, max_relative = 1e-6);
            }
        }
    }

    /// `theta = 0` with uniform weights must recover the plain (untilted)
    /// Hashimoto radius from the Bass-reduced `2n`-dim path, confirming the
    /// darc-space (`2m`-dim) construction lands on the same physical `B`.
    #[test]
    fn tilted_radius_at_theta_zero_matches_untilted_estimate() {
        let g = random_near_regular(30, 4, 7);
        assert!(g.is_connected());
        let darcs = DarcIndex::build(&g);
        let weights = ArcWeights::uniform(&darcs);

        let baseline = estimate_spectral_radius(&g, 40, 3);
        let tilted = tilted_spectral_radius(&darcs, &weights, 0.0, 40, 3);

        assert_relative_eq!(tilted.rho, baseline, epsilon = 1e-6, max_relative = 1e-4);
    }

    /// Uniform tilt (`f = 1` on every arc) has a closed form: `B(theta) =
    /// e^theta * B`, so `rho(theta) = e^theta * rho(0)` and, since
    /// `d/dtheta log rho = 1` identically, `drho/dtheta = rho(theta)`. This
    /// exercises the general left/right-eigenvector machinery against an
    /// answer that doesn't depend on it.
    #[test]
    fn tilted_radius_matches_uniform_closed_form() {
        let g = random_near_regular(24, 3, 11);
        assert!(g.is_connected());
        let darcs = DarcIndex::build(&g);
        let weights = ArcWeights::uniform(&darcs);
        let rho0 = estimate_spectral_radius(&g, 40, 5);

        for &theta in &[-0.5, 0.0, 0.3, 1.0] {
            let t = tilted_spectral_radius(&darcs, &weights, theta, 40, 5);
            let expected_rho = theta.exp() * rho0;
            assert_relative_eq!(t.rho, expected_rho, epsilon = 1e-5, max_relative = 1e-3);
            assert_relative_eq!(t.drho_dtheta, t.rho, epsilon = 1e-5, max_relative = 1e-3);
        }
    }

    /// For a non-uniform (vertex-potential) tilt with no closed form, the
    /// analytic `drho/dtheta` from `tilted_spectral_radius` must agree with
    /// a central finite difference of `rho(theta)` itself.
    #[test]
    fn tilted_radius_derivative_matches_finite_difference() {
        let g = random_near_regular(28, 4, 13);
        assert!(g.is_connected());
        let darcs = DarcIndex::build(&g);

        // A non-constant potential (degree-based, then centered) so the
        // tilt is genuinely non-uniform.
        let degrees = g.degrees();
        let mean_deg: f64 = degrees.iter().sum::<f64>() / degrees.len() as f64;
        let phi: Vec<f64> = degrees.iter().map(|d| (d - mean_deg) * 0.1).collect();
        let weights = ArcWeights::from_head_potential(&darcs, &phi);

        let theta0 = 0.2;
        let h = 1e-4;
        let center = tilted_spectral_radius(&darcs, &weights, theta0, 50, 9);
        let plus = tilted_spectral_radius(&darcs, &weights, theta0 + h, 50, 9);
        let minus = tilted_spectral_radius(&darcs, &weights, theta0 - h, 50, 9);
        let finite_diff = (plus.rho - minus.rho) / (2.0 * h);

        assert_relative_eq!(center.drho_dtheta, finite_diff, epsilon = 1e-3, max_relative = 5e-3);
    }

    /// Uniform tilt has closed form `Lambda(theta) = theta + log(rho0)`
    /// (since `rho(theta) = e^theta * rho0`), so `Lambda'(theta) = 1`
    /// identically: `x=1` is the *only* finite-rate point (every other `x`
    /// has `I(x) = +infinity`, since no `theta` can push the derivative off
    /// `1`), and there `I(1) = sup_theta(theta - (theta + log(rho0))) =
    /// -log(rho0)` for every `theta` alike -- i.e. the affine `Lambda`'s
    /// Legendre transform collapses to the single value `-log(rho0)` at
    /// `x=1` (not `0`, since `Lambda(0) = log(rho0) != 0` here -- this
    /// `rho_B(theta)` is a Perron root, not a properly-normalized MGF with
    /// `Lambda(0) = 0`).
    #[test]
    fn legendre_rate_matches_uniform_tilt_closed_form() {
        let g = random_near_regular(24, 3, 11);
        assert!(g.is_connected());
        let darcs = DarcIndex::build(&g);
        let weights = ArcWeights::uniform(&darcs);
        let rho0 = estimate_spectral_radius(&g, 40, 5);

        let point = legendre_rate(&darcs, &weights, 1.0, 40, 5);
        assert_relative_eq!(point.rate, -rho0.ln(), epsilon = 1e-4, max_relative = 1e-3);
    }

    /// For a genuinely non-uniform tilt, `I(x)` at `x = Lambda'(0)` (the
    /// untilted mean of the observable) must equal `-Lambda(0)` -- the
    /// minimum of the convex conjugate, attained exactly where `theta=0`
    /// itself solves the first-order condition -- and `I(x)` must be
    /// strictly larger a bit away from that mean (strict convexity of
    /// `Lambda` away from the trivial uniform-tilt case). Offsets are kept
    /// small (`+-0.01`) and `theta` searched only out to a modest range:
    /// the tilted operator becomes numerically ill-conditioned at large
    /// `|theta|` (the arc-weight dynamic range `exp(theta * phi)` grows
    /// too fast for Arnoldi to resolve the Perron root reliably), so this
    /// test deliberately stays inside the well-conditioned regime rather
    /// than probing the true edge of `Lambda`'s domain.
    #[test]
    fn legendre_rate_is_minimized_at_the_mean() {
        // stochastic_block_model (not random_near_regular) so vertex
        // degrees genuinely vary -- the degree-potential observable below
        // would be identically zero on a regular graph, making every `x`
        // unreachable except the trivial `x=0` case.
        let (g, _) = stochastic_block_model(3, 6, 0.7, 0.15, 13);
        assert!(g.is_connected());
        let darcs = DarcIndex::build(&g);

        let degrees = g.degrees();
        let mean_deg: f64 = degrees.iter().sum::<f64>() / degrees.len() as f64;
        let phi: Vec<f64> = degrees.iter().map(|d| (d - mean_deg) * 0.1).collect();
        let weights = ArcWeights::from_head_potential(&darcs, &phi);

        let krylov_dim = 50;
        let seed = 9;
        let (lambda_0, mean_x) = scaled_cgf(&darcs, &weights, 0.0, krylov_dim, seed);

        let at_mean = legendre_rate(&darcs, &weights, mean_x, krylov_dim, seed);
        assert_relative_eq!(at_mean.theta_star, 0.0, epsilon = 1e-4, max_relative = 1.0);
        assert_relative_eq!(at_mean.rate, -lambda_0, epsilon = 1e-4, max_relative = 1e-3);

        for &offset in &[0.01, -0.01, 0.02, -0.02] {
            let point = legendre_rate(&darcs, &weights, mean_x + offset, krylov_dim, seed);
            assert!(
                point.rate > at_mean.rate + 1e-8,
                "I(x) should be strictly larger than I(mean) away from the mean at \
                 offset={offset}: I(mean)={}, I(x)={}",
                at_mean.rate,
                point.rate
            );
        }
    }

    /// Consistency check: at the `theta_star` returned for a given `x`, the
    /// closed-form `Lambda'(theta_star)` (the same `drho_dtheta / rho`
    /// `tilted_spectral_radius` already exposes) must reproduce `x` itself
    /// -- this is exactly the first-order condition `legendre_rate` solves,
    /// checked here against the independent `scaled_cgf` helper. Offsets
    /// are kept small for the same numerical-conditioning reason as
    /// `legendre_rate_is_minimized_at_the_mean` above.
    #[test]
    fn legendre_rate_theta_star_satisfies_first_order_condition() {
        let (g, _) = stochastic_block_model(3, 6, 0.7, 0.15, 21);
        assert!(g.is_connected());
        let darcs = DarcIndex::build(&g);

        let degrees = g.degrees();
        let mean_deg: f64 = degrees.iter().sum::<f64>() / degrees.len() as f64;
        let phi: Vec<f64> = degrees.iter().map(|d| (d - mean_deg) * 0.1).collect();
        let weights = ArcWeights::from_head_potential(&darcs, &phi);

        let krylov_dim = 50;
        let seed = 17;
        let (_, mean_x) = scaled_cgf(&darcs, &weights, 0.0, krylov_dim, seed);

        for &x in &[mean_x - 0.02, mean_x - 0.005, mean_x + 0.005, mean_x + 0.02] {
            let point = legendre_rate(&darcs, &weights, x, krylov_dim, seed);
            let (_, lambda_prime) = scaled_cgf(&darcs, &weights, point.theta_star, krylov_dim, seed);
            assert_relative_eq!(lambda_prime, x, epsilon = 1e-4, max_relative = 1e-3);
        }
    }

    #[test]
    fn tree_has_zero_nontrivial_hashimoto_spectrum() {
        // On a tree every non-backtracking walk terminates (no cycles), so
        // B is nilpotent: all eigenvalues are exactly 0. This is the
        // degenerate negative-control case referenced by `random_tree`.
        let t = random_tree(15, 2);
        let (_darcs, b) = dense_hashimoto_matrix(&t);
        let max_abs = b.complex_eigenvalues().iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).fold(0.0, f64::max);
        // A tolerance looser than machine epsilon: this is the
        // largest-modulus eigenvalue of a genuinely nilpotent 2m x 2m
        // matrix recovered via a general dense eigensolver, so some
        // floating-point residue is expected numerical noise, not a sign
        // the construction is wrong.
        assert!(max_abs < 1e-2, "tree Hashimoto spectral radius should be ~0, got {max_abs}");
    }
}
