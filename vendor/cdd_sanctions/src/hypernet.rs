//! Spectral hypergraph analysis of the customer network.
//!
//! The CDD layer already resolves *ownership* exactly (effective percentages through any
//! number of layers, the 50% rule). What a pairwise ownership walk cannot see is
//! **association**: a customer that shares a director with a company owned by a sanctioned
//! person, co-owners that sit in the same structure, or two "unrelated" parties that quote
//! the same passport or registration number. Those are many-to-many facts, so the customer
//! base is modelled as a **hypergraph** (via the `spectral_hypergraph` crate):
//!
//! | hyperedge            | members                                            |
//! |----------------------|----------------------------------------------------|
//! | `Owners(entity)`     | the entity and every recorded owner                |
//! | `Control(entity)`    | the entity and its directors/controllers/signatories/trustees/settlors/beneficiaries |
//! | `Stake(owner→entity)`| the pair, weighted by the ownership percentage     |
//! | `SharedIdentifier`   | every party quoting the same identifier            |
//!
//! Two analyses run on it:
//!
//! 1. **Indirect sanctions exposure.** A random walk with restart (personalised PageRank on
//!    the hypergraph, Zhou–Huang–Schölkopf 2006) from sanctioned *seeds*. With
//!    `Θ = D⁻¹ᐟ² H W Dₑ⁻¹ Hᵀ D⁻¹ᐟ²` and damping `α`, the walk distribution for seed `s` is
//!    `p = D¹ᐟ² g` where `(I − αΘ) g = (1−α) D⁻¹ᐟ² e_s`. The system matrix is
//!    `(1−α)I + αΔ` (Δ = the crate's normalised Laplacian), symmetric positive definite with
//!    eigenvalues in `[1−α, 1]`, so conjugate gradients converge in a few dozen
//!    matrix-free products. `p` is a probability distribution: an exposure of `0.10` means
//!    10% of the walk's mass lands on that party. Mass is conserved, so a seed's influence is
//!    shared out, not amplified, and a hub with hundreds of members dilutes it.
//! 2. **Structure.** Per connected component: the algebraic connectivity λ₂ and the sparsest
//!    normalised cut found by a Cheeger sweep over the Fiedler vector. A low-conductance cut
//!    means two groups joined only through a few *connector* parties (typical of nominee
//!    arrangements). Reported, never scored.
//!
//! Everything here is **advisory**. Exposure feeds risk points and a tier floor (so approval
//! needs the MLRO); it never confirms a match or blocks a party by itself, and seeds that the
//! ownership rules already cover are excluded by the engine to avoid double counting.

use crate::cdd::{Party, PartyId, RelKind, Relationship};
use crate::error::{Error, Result};
use crate::links::{LinkKey, LinkKind, LinkPolicies};
use crate::model::IdKind;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use spectral_hypergraph::hypergraph::HypergraphBuilder;
use spectral_hypergraph::laplacian::{dense_normalized_laplacian, HypergraphOperator};
use spectral_hypergraph::operator::LinearOperator;
use spectral_hypergraph::spectral::{dense_eigen, lanczos_smallest};
use spectral_hypergraph::{SpectralHypergraph, VertexId};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

// ----------------------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NetworkConfig {
    /// Master switch for the exposure factor in risk rating. (The analysis functions still work.)
    pub enabled: bool,
    /// Random-walk continuation probability α in (0,1). Higher = influence travels further.
    pub damping: f64,
    pub owners_weight: f64,
    pub control_weight: f64,
    /// Multiplied by `percent/100` for each direct ownership stake.
    pub stake_weight: f64,
    pub shared_identifier_weight: f64,
    /// An identifier quoted by more parties than this is treated as junk data, not a link.
    pub max_shared_identifier_group: usize,
    /// Weight and group-size cap for each kind of shared address / phone / email / bank
    /// account / registered agent (`Party.links`).
    pub links: LinkPolicies,
    /// Digits of the home country's dial code (e.g. `"501"`), so a local phone number and
    /// the same number in international form are recognised as one. Off by default.
    pub home_dial_code: Option<String>,
    /// Seed strength of a confirmed match / blocked party.
    pub confirmed_seed: f64,
    /// Seed strength of a party with an unresolved high-confidence alert (0 disables).
    pub high_alert_seed: f64,
    /// Seed strength of a party with an unresolved review-level alert (0 disables).
    pub review_alert_seed: f64,
    /// Exposure (share of walk mass) at which the moderate factor and a Medium tier floor apply.
    /// Starting points calibrated on a synthetic network only: measure them on your own data.
    pub moderate_exposure: f64,
    /// Exposure at which the strong factor and a High tier floor apply.
    pub strong_exposure: f64,
    /// Cap on seeds used in portfolio-wide analysis (strongest first).
    pub max_seeds: usize,
    /// Components at least this large get a Cheeger-sweep split.
    pub split_min_size: usize,
    /// Components up to this size use the dense eigensolver; larger ones use Lanczos.
    pub dense_limit: usize,
    /// Components larger than this skip the structure analysis (Lanczos on ~66k parties
    /// took ~8 s); exposure is unaffected.
    pub structure_max_size: usize,
    pub solver_tolerance: f64,
    pub solver_max_iterations: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            damping: 0.6,
            owners_weight: 1.0,
            control_weight: 1.0,
            stake_weight: 1.0,
            shared_identifier_weight: 1.0,
            max_shared_identifier_group: 10,
            links: LinkPolicies::default(),
            home_dial_code: None,
            confirmed_seed: 1.0,
            high_alert_seed: 0.3,
            review_alert_seed: 0.0,
            moderate_exposure: 0.001,
            strong_exposure: 0.01,
            max_seeds: 256,
            split_min_size: 6,
            dense_limit: 400,
            structure_max_size: 25_000,
            solver_tolerance: 1e-10,
            solver_max_iterations: 500,
        }
    }
}

impl NetworkConfig {
    /// A copy with one edge family switched off (weight 0), or `None` for an unknown family name.
    pub fn without_family(&self, family: &str) -> Option<NetworkConfig> {
        let mut c = self.clone();
        match family {
            "ownership" => c.owners_weight = 0.0,
            "control" => c.control_weight = 0.0,
            "stake" => c.stake_weight = 0.0,
            "shared_identifier" => c.shared_identifier_weight = 0.0,
            other => c.links.set_weight(LinkKind::ALL.into_iter().find(|k| k.label() == other)?, 0.0),
        }
        Some(c)
    }

    fn alpha(&self) -> f64 {
        self.damping.clamp(0.05, 0.95)
    }
}

// ----------------------------------------------------------------------------------------
// Hyperedges
// ----------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Owners { entity: PartyId },
    Control { entity: PartyId },
    Stake { owner: PartyId, entity: PartyId, percent: f64 },
    /// `tail` is the last three characters of the shared value (enough to recognise it, not to leak it).
    SharedIdentifier { kind: IdKind, tail: String },
    /// Parties quoting the same address, phone, email, bank account or registered agent.
    /// `hint` is a masked rendering (see [`LinkKey::hint`]).
    SharedLink { kind: LinkKind, hint: String },
}

impl EdgeKind {
    /// Name of the edge family, as used by calibration (`cddctl calibrate`).
    pub fn family(&self) -> &'static str {
        match self {
            EdgeKind::Owners { .. } => "ownership",
            EdgeKind::Control { .. } => "control",
            EdgeKind::Stake { .. } => "stake",
            EdgeKind::SharedIdentifier { .. } => "shared_identifier",
            EdgeKind::SharedLink { kind, .. } => kind.label(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeSpec {
    pub kind: EdgeKind,
    pub members: Vec<PartyId>,
    pub weight: f64,
}

const JUNK_IDENTIFIERS: [&str; 8] = ["NONE", "NULL", "UNKNOWN", "NOTAVAILABLE", "NOTAPPLICABLE", "TEST", "TBD", "PENDING"];

/// Builds the hyperedge list from the CDD data. Returns the edges and the number of
/// shared-identifier groups skipped as implausibly large.
pub fn edge_specs(parties: &BTreeMap<PartyId, Party>, rels: &[Relationship], cfg: &NetworkConfig) -> (Vec<EdgeSpec>, usize) {
    let mut specs: Vec<EdgeSpec> = Vec::new();
    let mut owners: BTreeMap<PartyId, BTreeSet<PartyId>> = BTreeMap::new();
    let mut control: BTreeMap<PartyId, BTreeSet<PartyId>> = BTreeMap::new();
    let usable = |w: f64| w.is_finite() && w > 0.0;

    for r in rels {
        if r.from == r.to || !parties.contains_key(&r.from) || !parties.contains_key(&r.to) {
            continue;
        }
        if r.kind == RelKind::Owner {
            owners.entry(r.to).or_default().insert(r.from);
            if let Some(pct) = r.percent {
                let w = cfg.stake_weight * pct / 100.0;
                if usable(w) {
                    specs.push(EdgeSpec {
                        kind: EdgeKind::Stake { owner: r.from, entity: r.to, percent: pct },
                        members: vec![r.from, r.to],
                        weight: w,
                    });
                }
            }
        } else {
            control.entry(r.to).or_default().insert(r.from);
        }
    }
    if usable(cfg.owners_weight) {
        for (entity, set) in owners {
            let mut members = vec![entity];
            members.extend(set);
            specs.push(EdgeSpec { kind: EdgeKind::Owners { entity }, members, weight: cfg.owners_weight });
        }
    }
    if usable(cfg.control_weight) {
        for (entity, set) in control {
            let mut members = vec![entity];
            members.extend(set);
            specs.push(EdgeSpec { kind: EdgeKind::Control { entity }, members, weight: cfg.control_weight });
        }
    }

    let mut skipped = 0;
    if usable(cfg.shared_identifier_weight) {
        let mut groups: BTreeMap<(String, String, Option<String>), (IdKind, BTreeSet<PartyId>)> = BTreeMap::new();
        for (pid, p) in parties {
            for id in &p.identifiers {
                let key = id.key();
                let first = key.chars().next();
                if key.chars().count() < 4 || JUNK_IDENTIFIERS.contains(&key.as_str()) || key.chars().all(|c| Some(c) == first) {
                    continue;
                }
                let country = if id.kind.globally_unique() { None } else { id.country.clone() };
                groups.entry((format!("{:?}", id.kind), key, country)).or_insert_with(|| (id.kind, BTreeSet::new())).1.insert(*pid);
            }
        }
        for ((_, key, _), (kind, set)) in groups {
            if set.len() < 2 {
                continue;
            }
            if set.len() > cfg.max_shared_identifier_group {
                skipped += 1;
                continue;
            }
            let tail: String = key.chars().rev().take(3).collect::<Vec<_>>().into_iter().rev().collect();
            specs.push(EdgeSpec {
                kind: EdgeKind::SharedIdentifier { kind, tail },
                members: set.into_iter().collect(),
                weight: cfg.shared_identifier_weight,
            });
        }
    }

    // Addresses, phones, emails, bank accounts, registered agents.
    let mut linked: BTreeMap<LinkKey, BTreeSet<PartyId>> = BTreeMap::new();
    for (pid, p) in parties {
        for l in &p.links {
            if !usable(cfg.links.get(l.kind).weight) {
                continue;
            }
            if let Some(key) = l.key(cfg.home_dial_code.as_deref()) {
                linked.entry(key).or_default().insert(*pid);
            }
        }
    }
    for (key, set) in linked {
        if set.len() < 2 {
            continue;
        }
        let policy = cfg.links.get(key.kind);
        if set.len() > policy.max_group {
            skipped += 1;
            continue;
        }
        specs.push(EdgeSpec { kind: EdgeKind::SharedLink { kind: key.kind, hint: key.hint() }, members: set.into_iter().collect(), weight: policy.weight });
    }
    (specs, skipped)
}

fn describe_edge(kind: &EdgeKind, parties: &BTreeMap<PartyId, Party>) -> String {
    let nm = |id: &PartyId| parties.get(id).map(|p| format!("{} (#{id})", p.name)).unwrap_or_else(|| format!("#{id}"));
    match kind {
        EdgeKind::Owners { entity } => format!("ownership group of {}", nm(entity)),
        EdgeKind::Control { entity } => format!("directors/controllers of {}", nm(entity)),
        EdgeKind::Stake { owner, entity, percent } => format!("{:.0}% stake {} -> {}", percent, nm(owner), nm(entity)),
        EdgeKind::SharedIdentifier { kind, tail } => format!("shared {kind:?} identifier ending {tail}"),
        EdgeKind::SharedLink { kind, hint } => format!("shared {} ({hint})", kind.label()),
    }
}

// ----------------------------------------------------------------------------------------
// Connected components as spectral hypergraphs
// ----------------------------------------------------------------------------------------

/// One connected component. The spectral routines assume a connected hypergraph (a single
/// zero eigenvalue), so analysis is always per component.
pub struct ComponentGraph {
    parties: Vec<PartyId>,
    index: HashMap<PartyId, usize>,
    edges: Vec<EdgeSpec>,
    members: Vec<Vec<usize>>,
    incident: Vec<Vec<usize>>,
    degree: Vec<f64>,
    hg: SpectralHypergraph,
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Splits `specs` into connected components (ordered by smallest member id). With `only`,
/// just the component containing that party is built (none if the party has no links).
pub fn components(specs: Vec<EdgeSpec>, only: Option<PartyId>) -> Result<Vec<ComponentGraph>> {
    let specs: Vec<EdgeSpec> = specs
        .into_iter()
        .filter(|s| s.weight.is_finite() && s.weight > 0.0 && s.members.iter().collect::<BTreeSet<_>>().len() >= 2)
        .collect();
    let ids: BTreeSet<PartyId> = specs.iter().flat_map(|s| s.members.iter().copied()).collect();
    let idx: HashMap<PartyId, usize> = ids.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let mut parent: Vec<usize> = (0..ids.len()).collect();
    for s in &specs {
        let first = idx[&s.members[0]];
        for m in &s.members[1..] {
            let a = find(&mut parent, first);
            let b = find(&mut parent, idx[m]);
            if a != b {
                parent[b] = a;
            }
        }
    }
    let wanted = match only {
        Some(p) => match idx.get(&p) {
            Some(&i) => Some(find(&mut parent, i)),
            None => return Ok(vec![]),
        },
        None => None,
    };
    let mut groups: BTreeMap<usize, Vec<EdgeSpec>> = BTreeMap::new();
    for s in specs {
        let r = find(&mut parent, idx[&s.members[0]]);
        if wanted.map_or(true, |w| w == r) {
            groups.entry(r).or_default().push(s);
        }
    }
    let mut out = Vec::with_capacity(groups.len());
    for (_, g) in groups {
        out.push(ComponentGraph::build(g)?);
    }
    out.sort_by_key(|c| c.parties[0]);
    Ok(out)
}

impl ComponentGraph {
    fn build(specs: Vec<EdgeSpec>) -> Result<Self> {
        let ids: BTreeSet<PartyId> = specs.iter().flat_map(|s| s.members.iter().copied()).collect();
        let parties: Vec<PartyId> = ids.into_iter().collect();
        let index: HashMap<PartyId, usize> = parties.iter().enumerate().map(|(i, p)| (*p, i)).collect();
        let n = parties.len();
        let mut b = HypergraphBuilder::with_capacity(n, specs.len(), 3);
        for p in &parties {
            b.add_vertex(p.to_string())?;
        }
        let (mut members, mut incident, mut degree) = (Vec::new(), vec![Vec::new(); n], vec![0.0; n]);
        for (e, s) in specs.iter().enumerate() {
            let mut m: Vec<usize> = s.members.iter().map(|p| index[p]).collect();
            m.sort_unstable();
            m.dedup();
            let vids: Vec<VertexId> = m.iter().map(|&i| VertexId(i)).collect();
            b.add_hyperedge(&vids, s.weight)?;
            for &i in &m {
                degree[i] += s.weight;
                incident[i].push(e);
            }
            members.push(m);
        }
        Ok(Self { parties, index, edges: specs, members, incident, degree, hg: b.build()? })
    }

    pub fn parties(&self) -> &[PartyId] {
        &self.parties
    }
    pub fn len(&self) -> usize {
        self.parties.len()
    }
    pub fn is_empty(&self) -> bool {
        self.parties.is_empty()
    }
    pub fn edges(&self) -> &[EdgeSpec] {
        &self.edges
    }
    pub fn contains(&self, p: PartyId) -> bool {
        self.index.contains_key(&p)
    }
    /// The underlying `spectral_hypergraph` structure.
    pub fn hypergraph(&self) -> &SpectralHypergraph {
        &self.hg
    }
    /// Weighted degree of a party in this component.
    pub fn degree_of(&self, p: PartyId) -> Option<f64> {
        self.index.get(&p).map(|&i| self.degree[i])
    }

    fn operator(&self) -> Result<HypergraphOperator> {
        Ok(HypergraphOperator::new(&self.hg)?)
    }

    /// Solves `((1-α)I + αΔ) x = b` by conjugate gradients, matrix-free.
    fn solve(&self, op: &HypergraphOperator, b: &DVector<f64>, cfg: &NetworkConfig) -> Result<DVector<f64>> {
        let a = cfg.alpha();
        let n = b.len();
        let mut x = DVector::<f64>::zeros(n);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rs = r.dot(&r);
        let target = cfg.solver_tolerance * b.norm().max(f64::MIN_POSITIVE);
        for _ in 0..cfg.solver_max_iterations {
            if rs.sqrt() <= target {
                return Ok(x);
            }
            let ap = &p * (1.0 - a) + op.apply(&p) * a;
            let denom = p.dot(&ap);
            if !(denom > 0.0) {
                break;
            }
            let step = rs / denom;
            x += &p * step;
            r -= &ap * step;
            let rs_new = r.dot(&r);
            p = &r + &p * (rs_new / rs);
            rs = rs_new;
        }
        if rs.sqrt() <= target {
            Ok(x)
        } else {
            Err(Error::Spectral(format!("exposure solver did not converge (residual {:.3e})", rs.sqrt())))
        }
    }

    /// Full random-walk-with-restart distribution from seed vertex `si` (sums to 1).
    fn walk_from(&self, op: &HypergraphOperator, si: usize, cfg: &NetworkConfig) -> Result<DVector<f64>> {
        let mut b = DVector::<f64>::zeros(self.len());
        b[si] = (1.0 - cfg.alpha()) / self.degree[si].sqrt();
        let g = self.solve(op, &b, cfg)?;
        Ok(DVector::from_fn(self.len(), |v, _| self.degree[v].sqrt() * g[v]))
    }

    /// Exposure of one party to the given seeds, with a single solve (uses the symmetry
    /// `p⁽ˢ⁾[t] = √(d_t/d_s) · K[s,t]`). The party's own seed weight is ignored.
    /// Returns `(exposure, strongest contributing seed)`.
    pub fn exposure_at(&self, target: PartyId, seeds: &[(PartyId, f64)], cfg: &NetworkConfig) -> Result<(f64, Option<PartyId>)> {
        let Some(&t) = self.index.get(&target) else { return Ok((0.0, None)) };
        let op = self.operator()?;
        let mut b = DVector::<f64>::zeros(self.len());
        b[t] = 1.0 - cfg.alpha();
        let g = self.solve(&op, &b, cfg)?;
        let (mut total, mut best): (f64, Option<(PartyId, f64)>) = (0.0, None);
        for &(s, y) in seeds {
            let Some(&si) = self.index.get(&s) else { continue };
            if si == t || !(y > 0.0) {
                continue;
            }
            let c = y * (self.degree[t] / self.degree[si]).sqrt() * g[si];
            total += c;
            if best.map_or(true, |(_, bc)| c > bc) {
                best = Some((s, c));
            }
        }
        Ok((total, best.map(|(s, _)| s)))
    }

    /// Exposure of every party to the seeds (one solve per seed). A seed's own mass at
    /// itself is excluded. Returns per-party `(exposure, strongest contributing seed)`.
    pub fn exposure_all(&self, seeds: &[(PartyId, f64)], cfg: &NetworkConfig) -> Result<Vec<(f64, Option<PartyId>)>> {
        let n = self.len();
        let mut total = vec![0.0; n];
        let mut best: Vec<Option<(PartyId, f64)>> = vec![None; n];
        let op = self.operator()?;
        for &(s, y) in seeds {
            let Some(&si) = self.index.get(&s) else { continue };
            if !(y > 0.0) {
                continue;
            }
            let p = self.walk_from(&op, si, cfg)?;
            for v in 0..n {
                if v == si {
                    continue;
                }
                let c = y * p[v];
                total[v] += c;
                if best[v].map_or(true, |(_, bc)| c > bc) {
                    best[v] = Some((s, c));
                }
            }
        }
        Ok(total.into_iter().zip(best).map(|(t, b)| (t, b.map(|(s, _)| s))).collect())
    }

    /// Breadth-first search over the party/hyperedge incidence structure from `from`.
    /// `dist` counts hyperedges traversed; `via[v] = (previous vertex, hyperedge)`.
    fn bfs(&self, from: usize) -> (Vec<Option<usize>>, Vec<Option<(usize, usize)>>) {
        let n = self.len();
        let mut dist = vec![None; n];
        let mut via = vec![None; n];
        let mut seen_edge = vec![false; self.edges.len()];
        dist[from] = Some(0);
        let mut q = VecDeque::from([from]);
        while let Some(u) = q.pop_front() {
            let du = dist[u].unwrap_or(0);
            for &e in &self.incident[u] {
                if std::mem::replace(&mut seen_edge[e], true) {
                    continue;
                }
                for &v in &self.members[e] {
                    if dist[v].is_none() {
                        dist[v] = Some(du + 1);
                        via[v] = Some((u, e));
                        q.push_back(v);
                    }
                }
            }
        }
        (dist, via)
    }

    /// Renders the path to vertex `t` out of a BFS tree (`via`), one line per step.
    fn render_path(&self, via: &[Option<(usize, usize)>], t: usize, parties: &BTreeMap<PartyId, Party>) -> Vec<String> {
        let nm = |i: usize| {
            let id = self.parties[i];
            parties.get(&id).map(|p| format!("{} (#{id})", p.name)).unwrap_or_else(|| format!("#{id}"))
        };
        let mut steps = Vec::new();
        let mut cur = t;
        while let Some((prev, e)) = via[cur] {
            steps.push(format!("{} --[{}]--> {}", nm(prev), describe_edge(&self.edges[e].kind, parties), nm(cur)));
            cur = prev;
        }
        steps.reverse();
        steps
    }

    /// Human-readable shortest path `from -> to` (fewest hyperedges), as one line per step.
    pub fn explain_path(&self, from: PartyId, to: PartyId, parties: &BTreeMap<PartyId, Party>) -> Vec<String> {
        let (Some(&f), Some(&t)) = (self.index.get(&from), self.index.get(&to)) else { return vec![] };
        let (dist, via) = self.bfs(f);
        if dist[t].is_none() {
            return vec![];
        }
        self.render_path(&via, t, parties)
    }

    /// Smallest number of hyperedges between `from` and the nearest of `targets`.
    pub fn hops_to_nearest(&self, from: PartyId, targets: &[PartyId]) -> Option<usize> {
        let &f = self.index.get(&from)?;
        let (dist, _) = self.bfs(f);
        targets.iter().filter_map(|t| self.index.get(t).and_then(|&i| dist[i])).min()
    }

    // ---- structure ------------------------------------------------------------------

    /// Algebraic connectivity and the sparsest normalised cut (Cheeger sweep over the
    /// Fiedler vector). Components smaller than 3 parties have nothing to report.
    pub fn structure(&self, cfg: &NetworkConfig) -> Structure {
        let n = self.len();
        let mut out = Structure::default();
        if n < 3 {
            return out;
        }
        if n > cfg.structure_max_size {
            out.note = Some(format!("component has {n} parties, above structure_max_size ({}); structure analysis skipped", cfg.structure_max_size));
            return out;
        }
        let eig = if n <= cfg.dense_limit {
            dense_normalized_laplacian(&self.hg).map(|l| dense_eigen(&l))
        } else {
            HypergraphOperator::new(&self.hg).and_then(|op| lanczos_smallest(&op, 2, n.min(300).max(30), 1e-6, 0x5EC7_2A1))
        };
        let eig = match eig {
            Ok(e) => e,
            Err(e) => {
                out.note = Some(format!("eigen-decomposition unavailable: {e}"));
                return out;
            }
        };
        out.algebraic_connectivity = Some(eig.eigenvalues[1].max(0.0));
        if n < cfg.split_min_size {
            return out;
        }
        let u = eig.eigenvectors.column(1);
        let x: Vec<f64> = (0..n).map(|i| u[i] / self.degree[i].sqrt()).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));

        let vol_total: f64 = self.degree.iter().sum();
        let mut count = vec![0usize; self.edges.len()];
        let (mut cut, mut vol_s) = (0.0f64, 0.0f64);
        let (mut best_k, mut best_phi) = (0usize, f64::INFINITY);
        let f = |c: usize, delta: usize, w: f64| w * (c * (delta - c)) as f64 / delta as f64;
        for (k, &v) in order.iter().enumerate().take(n - 1) {
            for &e in &self.incident[v] {
                let delta = self.members[e].len();
                cut += f(count[e] + 1, delta, self.edges[e].weight) - f(count[e], delta, self.edges[e].weight);
                count[e] += 1;
            }
            vol_s += self.degree[v];
            let phi = cut / vol_s.min(vol_total - vol_s);
            if phi < best_phi - 1e-12 {
                best_phi = phi;
                best_k = k + 1;
            }
        }
        if best_k == 0 {
            return out;
        }
        let mut in_a = vec![false; n];
        for &v in &order[..best_k] {
            in_a[v] = true;
        }
        let (mut pos, mut neg) = (vec![false; n], vec![false; n]);
        for (e, m) in self.members.iter().enumerate() {
            let a = m.iter().filter(|&&v| in_a[v]).count();
            let b = m.len() - a;
            let tgt = if a > b { Some(&mut pos) } else if b > a { Some(&mut neg) } else { None };
            if let Some(t) = tgt {
                for &v in &self.members[e] {
                    t[v] = true;
                }
            }
        }
        let mut side_a: Vec<PartyId> = Vec::new();
        let mut side_b: Vec<PartyId> = Vec::new();
        let mut connectors: Vec<PartyId> = Vec::new();
        for v in 0..n {
            (if in_a[v] { &mut side_a } else { &mut side_b }).push(self.parties[v]);
            if pos[v] && neg[v] {
                connectors.push(self.parties[v]);
            }
        }
        out.split = Some(Split { side_a, side_b, conductance: best_phi, connectors });
        out
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Structure {
    /// λ₂ of the normalised hypergraph Laplacian: near 0 = loosely joined, larger = cohesive.
    pub algebraic_connectivity: Option<f64>,
    pub split: Option<Split>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Split {
    pub side_a: Vec<PartyId>,
    pub side_b: Vec<PartyId>,
    /// cut weight / smaller side's volume; small = a weak link between the two sides.
    pub conductance: f64,
    /// Parties that belong to groups (hyperedges) on both sides of the cut.
    pub connectors: Vec<PartyId>,
}

// ----------------------------------------------------------------------------------------
// Reports
// ----------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartyExposure {
    pub party: PartyId,
    pub exposure: f64,
    /// The seed that contributes most.
    pub source: Option<PartyId>,
    pub hops: Option<usize>,
    /// Shortest connection from the source to this party.
    pub path: Vec<String>,
    /// Set by the engine: the source already sits in this party's ownership/control network,
    /// so the ownership rules cover it and it is not counted again in risk rating.
    pub covered_by_ownership_rules: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentReport {
    /// Smallest party id in the component.
    pub id: PartyId,
    pub parties: Vec<PartyId>,
    pub hyperedges: usize,
    pub seeds: Vec<(PartyId, f64)>,
    pub structure: Structure,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NetworkAnalysis {
    /// Components with at least 3 parties or at least one seed.
    pub components: Vec<ComponentReport>,
    /// Non-zero exposures, highest first.
    pub exposures: Vec<PartyExposure>,
    pub skipped_shared_identifier_groups: usize,
    pub seeds_truncated: bool,
}

/// Portfolio-wide analysis. `seeds` are `(party, strength)`; the strongest `max_seeds` are used.
pub fn analyze(parties: &BTreeMap<PartyId, Party>, rels: &[Relationship], seeds: &[(PartyId, f64)], cfg: &NetworkConfig) -> Result<NetworkAnalysis> {
    let (specs, skipped) = edge_specs(parties, rels, cfg);
    let mut seeds: Vec<(PartyId, f64)> = seeds.iter().copied().filter(|(_, y)| *y > 0.0).collect();
    seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    let seeds_truncated = seeds.len() > cfg.max_seeds;
    seeds.truncate(cfg.max_seeds);

    let mut rep = NetworkAnalysis { skipped_shared_identifier_groups: skipped, seeds_truncated, ..Default::default() };
    for comp in components(specs, None)? {
        let local: Vec<(PartyId, f64)> = seeds.iter().copied().filter(|(s, _)| comp.contains(*s)).collect();
        if comp.len() < 3 && local.is_empty() {
            continue;
        }
        if !local.is_empty() {
            let exp = comp.exposure_all(&local, cfg)?;
            // one BFS tree per distinct source, shared by every party it reaches
            type Tree = (Vec<Option<usize>>, Vec<Option<(usize, usize)>>);
            let mut cache: HashMap<PartyId, Tree> = HashMap::new();
            for (i, (e, src)) in exp.into_iter().enumerate() {
                if e <= 0.0 {
                    continue;
                }
                let party = comp.parties[i];
                let (hops, path) = match src {
                    Some(s) => {
                        let (dist, via) = cache.entry(s).or_insert_with(|| comp.bfs(comp.index[&s]));
                        (dist[i], if dist[i].is_some() { comp.render_path(via, i, parties) } else { vec![] })
                    }
                    None => (None, vec![]),
                };
                rep.exposures.push(PartyExposure { party, exposure: e, source: src, hops, path, covered_by_ownership_rules: false });
            }
        }
        rep.components.push(ComponentReport {
            id: comp.parties[0],
            parties: comp.parties.clone(),
            hyperedges: comp.edges.len(),
            seeds: local,
            structure: comp.structure(cfg),
        });
    }
    rep.exposures.sort_by(|a, b| b.exposure.partial_cmp(&a.exposure).unwrap_or(std::cmp::Ordering::Equal).then(a.party.cmp(&b.party)));
    Ok(rep)
}

/// Exposure of a single party. `seeds` must already exclude anything the caller does not
/// want counted (the engine removes the party itself and its ownership/control network).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RootExposure {
    pub exposure: f64,
    pub source: Option<PartyId>,
    pub hops: Option<usize>,
    pub path: Vec<String>,
}

pub fn exposure_for(
    parties: &BTreeMap<PartyId, Party>,
    rels: &[Relationship],
    seeds: &[(PartyId, f64)],
    root: PartyId,
    cfg: &NetworkConfig,
) -> Result<RootExposure> {
    if seeds.is_empty() {
        return Ok(RootExposure::default());
    }
    let (specs, _) = edge_specs(parties, rels, cfg);
    let comps = components(specs, Some(root))?;
    let Some(comp) = comps.first() else { return Ok(RootExposure::default()) };
    let local: Vec<(PartyId, f64)> = seeds.iter().copied().filter(|(s, y)| *s != root && *y > 0.0 && comp.contains(*s)).collect();
    if local.is_empty() {
        return Ok(RootExposure::default());
    }
    let (exposure, source) = comp.exposure_at(root, &local, cfg)?;
    let ids: Vec<PartyId> = local.iter().map(|(s, _)| *s).collect();
    let hops = comp.hops_to_nearest(root, &ids);
    let path = source.map(|s| comp.explain_path(s, root, parties)).unwrap_or_default();
    Ok(RootExposure { exposure, source, hops, path })
}

// ----------------------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdd::PartyKind;
    use crate::model::Identifier;
    use nalgebra::DMatrix;

    fn cfg() -> NetworkConfig {
        NetworkConfig::default()
    }
    fn ctl(entity: PartyId, members: &[PartyId]) -> EdgeSpec {
        let mut m = vec![entity];
        m.extend_from_slice(members);
        EdgeSpec { kind: EdgeKind::Control { entity }, members: m, weight: 1.0 }
    }
    fn one(specs: Vec<EdgeSpec>) -> ComponentGraph {
        let mut c = components(specs, None).unwrap();
        assert_eq!(c.len(), 1);
        c.remove(0)
    }
    fn party(id: PartyId, kind: PartyKind) -> Party {
        let mut p = Party::new(kind, &format!("P{id}"));
        p.id = id;
        p
    }

    /// Independent dense reference: p = (1-α)(I - α Bᵀ D⁻¹)⁻¹ e_s built straight from the definition.
    fn dense_walk(comp: &ComponentGraph, si: usize, alpha: f64) -> DVector<f64> {
        let n = comp.len();
        let mut b = DMatrix::<f64>::zeros(n, n);
        for (e, m) in comp.members.iter().enumerate() {
            let w = comp.edges[e].weight / m.len() as f64;
            for &i in m {
                for &j in m {
                    b[(i, j)] += w;
                }
            }
        }
        let pt = DMatrix::from_fn(n, n, |i, j| b[(i, j)] / comp.degree[j]);
        let m = (DMatrix::<f64>::identity(n, n) - pt * alpha).try_inverse().expect("invertible") * (1.0 - alpha);
        m.column(si).into_owned()
    }

    fn ring_with_hub() -> ComponentGraph {
        // groups: {1,2,3}, {3,4}, {4,5,6,7}, {2,8}, {8,9}
        one(vec![ctl(1, &[2, 3]), ctl(3, &[4]), ctl(4, &[5, 6, 7]), ctl(2, &[8]), ctl(8, &[9])])
    }

    #[test]
    fn walk_matches_independent_dense_inverse_and_conserves_mass() {
        let comp = ring_with_hub();
        let c = cfg();
        let op = comp.operator().unwrap();
        for si in 0..comp.len() {
            let fast = comp.walk_from(&op, si, &c).unwrap();
            let slow = dense_walk(&comp, si, c.alpha());
            assert!((&fast - &slow).norm() < 1e-8, "seed {si} differs by {}", (&fast - &slow).norm());
            assert!((fast.sum() - 1.0).abs() < 1e-8, "mass {}", fast.sum());
            assert!(fast.iter().all(|x| *x >= -1e-12));
        }
    }

    #[test]
    fn single_solve_form_equals_per_seed_form() {
        let comp = ring_with_hub();
        let c = cfg();
        let seeds = vec![(1u64, 1.0), (7, 0.3), (9, 0.6)];
        let all = comp.exposure_all(&seeds, &c).unwrap();
        for (i, p) in comp.parties().iter().enumerate() {
            let (e, src) = comp.exposure_at(*p, &seeds, &c).unwrap();
            assert!((e - all[i].0).abs() < 1e-9, "party {p}: {e} vs {}", all[i].0);
            assert_eq!(src, all[i].1, "party {p}");
        }
    }

    #[test]
    fn exposure_decays_with_distance_and_ignores_other_components() {
        // chain of groups 1-2-3-4-5-6 plus a separate pair 20-21
        let comp_specs = vec![ctl(1, &[2]), ctl(2, &[3]), ctl(3, &[4]), ctl(4, &[5]), ctl(5, &[6]), ctl(20, &[21])];
        let comps = components(comp_specs, None).unwrap();
        assert_eq!(comps.len(), 2);
        let chain = &comps[0];
        let ex = chain.exposure_all(&[(1, 1.0)], &cfg()).unwrap();
        let v: Vec<f64> = (2..=6).map(|p| ex[chain.index[&p]].0).collect();
        assert!(v.windows(2).all(|w| w[0] > w[1]), "not monotone: {v:?}");
        assert_eq!(ex[chain.index[&1]].0, 0.0, "a seed's own mass is excluded");
        assert_eq!(comps[1].exposure_all(&[(1, 1.0)], &cfg()).unwrap().iter().map(|x| x.0).sum::<f64>(), 0.0);
    }

    #[test]
    fn larger_groups_dilute_influence() {
        // seed 1 in a pair with 2, versus seed 10 in a group of five
        let comps = components(vec![ctl(1, &[2]), ctl(10, &[11, 12, 13, 14])], None).unwrap();
        let small = comps[0].exposure_all(&[(1, 1.0)], &cfg()).unwrap()[comps[0].index[&2]].0;
        let big = comps[1].exposure_all(&[(10, 1.0)], &cfg()).unwrap()[comps[1].index[&11]].0;
        assert!(small > big * 2.0, "{small} vs {big}");
    }

    #[test]
    fn edge_specs_cover_ownership_control_and_shared_identifiers() {
        let mut ps: BTreeMap<PartyId, Party> = BTreeMap::new();
        for (id, k) in [(1, PartyKind::Entity), (2, PartyKind::Individual), (3, PartyKind::Individual), (4, PartyKind::Individual)] {
            ps.insert(id, party(id, k));
        }
        let rel = |id, from, to, kind, pct| Relationship { id, from, to, kind, percent: pct };
        let rels = vec![rel(1, 2, 1, RelKind::Owner, Some(60.0)), rel(2, 3, 1, RelKind::Director, None), rel(3, 99, 1, RelKind::Owner, Some(5.0))];
        let (specs, _) = edge_specs(&ps, &rels, &cfg());
        assert!(specs.iter().any(|s| matches!(s.kind, EdgeKind::Stake { owner: 2, entity: 1, .. }) && (s.weight - 0.6).abs() < 1e-12));
        assert!(specs.iter().any(|s| matches!(s.kind, EdgeKind::Owners { entity: 1 }) && s.members == vec![1, 2]));
        assert!(specs.iter().any(|s| matches!(s.kind, EdgeKind::Control { entity: 1 }) && s.members == vec![1, 3]));
        assert!(!specs.iter().any(|s| s.members.contains(&99)), "relationship to an unknown party is ignored");

        // shared passport links 3 and 4; the same value with a junk key or in 12 parties does not
        ps.get_mut(&3).unwrap().identifiers.push(Identifier::new(IdKind::Passport, "X-123 456", Some("BZ")));
        ps.get_mut(&4).unwrap().identifiers.push(Identifier::new(IdKind::Passport, "x123456", Some("Belize")));
        ps.get_mut(&1).unwrap().identifiers.push(Identifier::new(IdKind::TaxId, "N/A", None));
        ps.get_mut(&2).unwrap().identifiers.push(Identifier::new(IdKind::TaxId, "N/A", None));
        let (specs, skipped) = edge_specs(&ps, &rels, &cfg());
        let shared: Vec<&EdgeSpec> = specs.iter().filter(|s| matches!(s.kind, EdgeKind::SharedIdentifier { .. })).collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].members, vec![3, 4]);
        assert_eq!(skipped, 0);

        let mut many: BTreeMap<PartyId, Party> = BTreeMap::new();
        for id in 1..=12u64 {
            let mut p = party(id, PartyKind::Individual);
            p.identifiers.push(Identifier::new(IdKind::NationalId, "SAME-ID-9999", Some("BZ")));
            many.insert(id, p);
        }
        let (specs, skipped) = edge_specs(&many, &[], &cfg());
        assert!(specs.is_empty());
        assert_eq!(skipped, 1);
    }

    #[test]
    fn explain_path_and_hops() {
        let mut ps: BTreeMap<PartyId, Party> = BTreeMap::new();
        for id in 1..=4u64 {
            ps.insert(id, party(id, PartyKind::Individual));
        }
        let comp = one(vec![ctl(1, &[2]), ctl(2, &[3]), ctl(3, &[4])]);
        let path = comp.explain_path(1, 4, &ps);
        assert_eq!(path.len(), 3);
        assert!(path[0].starts_with("P1 (#1) --[directors/controllers of"));
        assert!(path[2].ends_with("--> P4 (#4)"));
        assert_eq!(comp.hops_to_nearest(1, &[4, 3]), Some(2));
        assert_eq!(comp.hops_to_nearest(1, &[99]), None);
    }

    #[test]
    fn sweep_cut_finds_the_bridge_between_two_clusters() {
        // two tightly knit groups {1..5} and {11..15}, joined only through person 50 who
        // sits in a group on each side
        let comp = one(vec![
            ctl(1, &[2, 3, 4, 5]),
            ctl(2, &[3, 4, 5]),
            ctl(11, &[12, 13, 14, 15]),
            ctl(12, &[13, 14, 15]),
            ctl(100, &[5, 50]),
            ctl(200, &[50, 11]),
        ]);
        let s = comp.structure(&cfg());
        let split = s.split.expect("split");
        assert!(split.conductance < 0.2, "conductance {}", split.conductance);
        let side = |p: PartyId| split.side_a.contains(&p);
        assert!((1..=5).all(|p| side(p) == side(1)), "left cluster split apart");
        assert!((11..=15).all(|p| side(p) == side(11)), "right cluster split apart");
        assert_ne!(side(1), side(11));
        assert!(split.connectors.contains(&50), "connectors {:?}", split.connectors);
        let tight = one(vec![ctl(1, &[2, 3, 4, 5, 6, 7])]).structure(&cfg());
        assert!(tight.algebraic_connectivity.unwrap() > s.algebraic_connectivity.unwrap());
    }

    #[test]
    fn dense_and_lanczos_agree_on_algebraic_connectivity() {
        let specs = vec![ctl(1, &[2, 3]), ctl(3, &[4, 5]), ctl(5, &[6, 7]), ctl(7, &[8, 9]), ctl(9, &[10, 1])];
        let comp = one(specs);
        let dense = comp.structure(&NetworkConfig { dense_limit: 1000, ..cfg() }).algebraic_connectivity.unwrap();
        let lanc = comp.structure(&NetworkConfig { dense_limit: 0, ..cfg() }).algebraic_connectivity.unwrap();
        assert!((dense - lanc).abs() < 1e-5, "{dense} vs {lanc}");
    }

    #[test]
    fn oversized_components_skip_structure_analysis_with_a_note() {
        let comp = one(vec![ctl(1, &[2, 3, 4]), ctl(4, &[5, 6, 7])]);
        let s = comp.structure(&NetworkConfig { structure_max_size: 5, ..cfg() });
        assert!(s.algebraic_connectivity.is_none() && s.split.is_none());
        assert!(s.note.unwrap().contains("structure_max_size"));
        assert!(comp.structure(&cfg()).algebraic_connectivity.is_some());
    }

    #[test]
    fn analyze_reports_components_and_ranked_exposures() {
        let mut ps: BTreeMap<PartyId, Party> = BTreeMap::new();
        for id in 1..=6u64 {
            ps.insert(id, party(id, if id <= 2 { PartyKind::Entity } else { PartyKind::Individual }));
        }
        let rel = |id, from, to, kind, pct| Relationship { id, from, to, kind, percent: pct };
        let rels = vec![
            rel(1, 3, 1, RelKind::Owner, Some(80.0)),
            rel(2, 4, 1, RelKind::Director, None),
            rel(3, 4, 2, RelKind::Director, None),
            rel(4, 5, 2, RelKind::Owner, Some(100.0)),
        ];
        // party 6 is isolated
        let rep = analyze(&ps, &rels, &[(3, 1.0)], &cfg()).unwrap();
        assert_eq!(rep.components.len(), 1);
        assert_eq!(rep.components[0].parties, vec![1, 2, 3, 4, 5]);
        assert_eq!(rep.components[0].seeds, vec![(3, 1.0)]);
        let order: Vec<PartyId> = rep.exposures.iter().map(|e| e.party).collect();
        assert_eq!(order[0], 1, "the directly owned company is most exposed: {order:?}");
        assert!(!order.contains(&6) && !order.contains(&3));
        let top = &rep.exposures[0];
        assert_eq!((top.source, top.hops), (Some(3), Some(1)));
        assert!(rep.exposures.windows(2).all(|w| w[0].exposure >= w[1].exposure));
        let e5 = rep.exposures.iter().find(|e| e.party == 5).unwrap();
        // 3 -> entity 1 -> director 4 -> entity 2 -> owner 5
        assert_eq!(e5.hops, Some(4));
        assert_eq!(e5.path.len(), 4);
    }
}
