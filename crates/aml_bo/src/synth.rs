//! Synthetic ownership world: a transaction stream plus a CDD register to match.
//!
//! **Not evidence about real money laundering.** It extends `aml_graph::synth` (typologies 1-7) with four
//! typologies that are defined by ownership as well as by flows (8-11), and with benign structures that look
//! similar on purpose (corporate groups moving money between subsidiaries, deep holding chains with known
//! owners, opaque-but-quiet nominee companies, near-miss sanctioned stakes, namesake false positives).
//! Because both the flows and the ownership were written by the same author, a model that finds them shows that the
//! machinery works, not how well it would do on a bank's data. Ownership is a static snapshot (as if the KYC
//! register were complete from day one), which is more generous than reality.

use crate::registry::AccountRegistry;
use aml_graph::synth::{generate, SynthConfig, DAY, T0};
use aml_graph::GTxn;
use aml_engine::{Country, PaymentType};
use cdd_sanctions::cdd::*;
use cdd_sanctions::engine::*;
use cdd_sanctions::model::*;
use chrono::{DateTime, TimeZone, Utc};
use hcrnn::rng::Rng;

pub const BO_TYPOLOGIES: [&str; 12] = [
    "(normal)", "Fan_In_Structuring", "Cycle", "Layered_Chain", "Slow_Drip", "Dormant_Pass_Through", "Bipartite_Mesh", "Behaviour_Ramp",
    "Shell_Group_Layering", "Sanctioned_Owner_Evasion", "Nominee_Pass_Through", "Split_Stake_Structuring",
];

#[derive(Clone, Debug)]
pub struct BoSynthConfig {
    pub base: SynthConfig,
    /// schemes per ownership typology (8-11)
    pub schemes: usize,
    /// benign look-alike structures per kind
    pub confounders: usize,
}
impl Default for BoSynthConfig {
    fn default() -> Self {
        BoSynthConfig { base: SynthConfig { days: 150, individuals: 1500, merchants: 60, schemes_per_type: 10, seed: 11 }, schemes: 12, confounders: 12 }
    }
}

pub struct OwnershipWorld {
    pub txns: Vec<GTxn>,
    pub compliance: Compliance,
    pub registry: AccountRegistry,
    pub typologies: Vec<String>,
}

const GIVEN: [&str; 12] = ["Ivan", "Dmitri", "Omar", "Viktor", "Karim", "Anton", "Rashid", "Pavel", "Nikolai", "Tariq", "Boris", "Sergei"];
const SYL_A: [&str; 10] = ["Pet", "Kov", "Vol", "Mor", "Sok", "Bel", "Gor", "Zub", "Lar", "Han"];
const SYL_B: [&str; 8] = ["ro", "ale", "ko", "ni", "ba", "ste", "mi", "lu"];
const HRJ: [&str; 4] = ["VG", "PA", "KY", "SC"];
const NORMAL_CC: [&str; 6] = ["GB", "GB", "US", "DE", "FR", "NL"];

fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, 1, 9, 0, 0).unwrap()
}
fn cc(c: &str) -> Country {
    Country::parse(c).unwrap()
}

struct Gen {
    c: Compliance,
    ana: Actor,
    mia: Actor,
    rng: Rng,
    reg: AccountRegistry,
    out: Vec<GTxn>,
    next_acct: i64,
    days: i64,
    merch: Vec<i64>,
    pool: Vec<i64>,
}

impl Gen {
    fn party(&mut self, kind: PartyKind, name: &str, countries: &[&str], nominee: bool) -> PartyId {
        let mut p = Party::new(kind, name);
        p.countries = countries.iter().map(|s| s.to_string()).collect();
        p.nominee_or_bearer_shares = nominee;
        self.c.add_party(p, &self.ana, t0()).expect("add party")
    }
    fn person(&mut self, name: &str) -> PartyId {
        let cn = NORMAL_CC[self.rng.below(NORMAL_CC.len())];
        self.party(PartyKind::Individual, name, &[cn], false)
    }
    fn entity(&mut self, name: &str, country: &str, nominee: bool) -> PartyId {
        self.party(PartyKind::Entity, name, &[country], nominee)
    }
    fn own(&mut self, from: PartyId, to: PartyId, pct: f64) {
        self.c.add_relationship(from, to, RelKind::Owner, Some(pct), &self.ana, t0()).expect("relationship");
    }
    fn account(&mut self, p: PartyId) -> i64 {
        self.next_acct += 1;
        self.reg.insert(self.next_acct, p);
        self.next_acct
    }
    fn hrj(&mut self) -> &'static str {
        HRJ[self.rng.below(HRJ.len())]
    }
    fn day(&mut self, lo: i64, extra: i64) -> i64 {
        lo + (self.rng.uniform() * (self.days - lo - extra).max(1) as f64) as i64
    }
    fn lognormal(&mut self, mu: f64, s: f64) -> f64 {
        (mu + s * self.rng.normal()).exp()
    }
    #[allow(clippy::too_many_arguments)]
    fn tx(&mut self, day: i64, s: i64, r: i64, amount: f64, pt: PaymentType, sc: &str, rc: &str, typ: u8) {
        let sec = (self.rng.uniform() * DAY as f64) as i64;
        let (a, b) = (cc(sc), cc(rc));
        let cross = a != b && self.rng.uniform() < 0.7;
        self.out.push(GTxn { ts: T0 + day * DAY + sec, sender: s, receiver: r, amount: amount.max(1.0), ptype: pt, sc: a, rc: b, cross_ccy: cross, rf: 0, label: typ > 0, typology: typ, rule: 0.0, rule_retro: 0.0 });
    }
    fn counterparty(&mut self) -> i64 {
        if self.rng.uniform() < 0.5 { self.merch[self.rng.below(self.merch.len())] } else { self.pool[self.rng.below(self.pool.len())] }
    }
    /// ordinary customer-like activity for an account: payments to and from merchants and individuals
    fn normal_activity(&mut self, acct: i64, from: i64, to: i64, rate: f64, mu: f64, home: &str) {
        for d in from..to {
            let l = (-rate).exp();
            let (mut k, mut p) = (0, 1.0);
            loop {
                p *= self.rng.uniform();
                if p <= l || k > 12 {
                    break;
                }
                k += 1;
            }
            for _ in 0..k {
                let other = self.counterparty();
                let amt = self.lognormal(mu, 0.8);
                let (s, r) = if self.rng.uniform() < 0.5 { (acct, other) } else { (other, acct) };
                let pt = if self.rng.uniform() < 0.5 { PaymentType::Ach } else { PaymentType::OnlineTransfer };
                self.tx(d, s, r, amt, pt, home, "GB", 0);
            }
        }
    }
    fn fresh_individual(&mut self) -> (i64, PartyId) {
        let n = self.reg.len();
        let p = self.person(&format!("Customer X{n}"));
        (self.account(p), p)
    }
    fn sanctioned_name(i: usize) -> (String, String, PartialDate) {
        let sur = format!("{}{}ov{}", SYL_A[i % SYL_A.len()], SYL_B[(i / SYL_A.len()) % SYL_B.len()], i / 80);
        let given = GIVEN[i % GIVEN.len()];
        (sur, given.to_string(), PartialDate::ymd(1958 + (i % 30) as i32, 1 + (i % 12) as u8, 1 + (i % 27) as u8))
    }
    /// A listed person who is a confirmed match in the register (real alert workflow: screen, then MLRO confirms).
    fn confirmed_sanctioned(&mut self, i: usize) -> PartyId {
        let (sur, given, dob) = Self::sanctioned_name(i);
        let mut p = Party::new(PartyKind::Individual, &format!("{given} {}", capitalise(&sur)));
        p.dob = vec![dob];
        p.nationalities = vec!["Russia".into()];
        let id = self.c.add_party(p, &self.ana, t0()).expect("sanctioned person");
        self.c.screen_party(id, &self.ana, t0()).expect("screen");
        let ids: Vec<u64> = self.c.alerts().values().filter(|a| a.party == id && a.status.is_unresolved()).map(|a| a.id).collect();
        for a in ids {
            let _ = self.c.resolve_alert(a, Outcome::ConfirmedMatch, "name, date of birth and nationality match the listing", &self.mia, t0());
        }
        id
    }
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

fn list_entry(i: usize) -> ListEntry {
    let (sur, given, dob) = Gen::sanctioned_name(i);
    ListEntry {
        list: "SYNTH".into(),
        uid: format!("S{i}"),
        kind: SubjectKind::Individual,
        names: vec![NameVariant { name: format!("{}, {given}", sur.to_uppercase()), strength: AliasStrength::Primary }],
        dobs: vec![dob],
        nationalities: vec!["RU".into()],
        countries: vec![],
        identifiers: vec![],
        programs: vec!["SYNTHPROG".into()],
        remarks: None,
        listed_on: None,
        content_hash: String::new(),
    }
    .finalize()
}

pub fn generate_world(cfg: &BoSynthConfig) -> OwnershipWorld {
    let base = generate(&cfg.base);
    let mut c = Compliance::default();
    for (k, v) in [("VG", 2u8), ("PA", 2), ("KY", 2), ("SC", 2)] {
        c.risk_cfg.country_risk.insert(k.into(), v);
    }
    let (ana, mia) = (Actor::new("ana", Role::Analyst), Actor::new("mia", Role::Mlro));
    let n_s = cfg.schemes;
    let n_c = cfg.confounders;
    // listed persons: n_s evasion schemes, n_c near misses, n_c namesakes
    let n_list = n_s + 2 * n_c;
    c.load_list("SYNTH", (0..n_list).map(list_entry).collect(), "v1", false, &mia, t0()).expect("load list");
    let mut g = Gen { c, ana, mia, rng: Rng::new(cfg.base.seed ^ 0xB0_B0), reg: AccountRegistry::new(), out: vec![], next_acct: 40_000_000, days: cfg.base.days, merch: vec![], pool: vec![] };

    // ---- base population
    let mut seen: std::collections::BTreeSet<i64> = Default::default();
    for t in &base {
        seen.insert(t.sender);
        seen.insert(t.receiver);
    }
    g.merch = seen.iter().copied().filter(|a| (20_000_000..21_000_000).contains(a)).collect();
    g.pool = seen.iter().copied().filter(|a| (10_000_000..11_000_000).contains(a)).collect();
    let widely_held = g.entity("Widely Held Holdings Plc", "GB", false); // listed company: no recorded owners, benign
    for &a in &seen {
        let p = if (21_000_000..23_000_000).contains(&a) {
            let e = g.entity(&format!("Payments Ltd {a}"), "GB", false);
            g.own(widely_held, e, 100.0);
            e
        } else if (20_000_000..21_000_000).contains(&a) {
            let e = g.entity(&format!("Merchant {a} Ltd"), "GB", false);
            let k = 1 + g.rng.below(3);
            let mut left = 100.0;
            for j in 0..k {
                let o = g.person(&format!("Owner M{a}-{j}"));
                let pct = if j + 1 == k { left } else { (left * (0.4 + 0.3 * g.rng.uniform())).round() };
                left -= pct;
                g.own(o, e, pct);
            }
            e
        } else if a >= 30_000_000 && g.rng.uniform() < 0.15 {
            // scheme accounts of typologies 1-7 sometimes sit in ordinary small companies
            let ncc = NORMAL_CC[g.rng.below(NORMAL_CC.len())];
            let e = g.entity(&format!("Trading Co {a}"), ncc, false);
            let o = g.person(&format!("Owner T{a}"));
            g.own(o, e, 100.0);
            e
        } else {
            g.person(&format!("Customer {a}"))
        };
        g.reg.insert(a, p);
    }
    for t in &base {
        g.out.push(t.clone());
    }

    // ---- benign look-alikes (label 0)
    for k in 0..n_c {
        // C1 legitimate corporate group: money moves between subsidiaries on a schedule
        let owners: Vec<PartyId> = (0..3).map(|j| g.person(&format!("Group owner {k}-{j}"))).collect();
        let hold = g.entity(&format!("Group {k} Holdings Ltd"), "GB", false);
        for (o, p) in owners.iter().zip([40.0, 35.0, 25.0]) {
            g.own(*o, hold, p);
        }
        let subs: Vec<(PartyId, i64)> = (0..(3 + g.rng.below(3)))
            .map(|j| {
                let e = g.entity(&format!("Group {k} Sub {j} Ltd"), "GB", false);
                g.own(hold, e, 100.0);
                (e, g.account(e))
            })
            .collect();
        let ha = g.account(hold);
        let start = 16 + g.rng.below(20) as i64;
        for d in (start..g.days).step_by(7) {
            for i in 0..subs.len() {
                if g.rng.uniform() < 0.5 {
                    let j = g.rng.below(subs.len());
                    if i != j {
                        let a = g.lognormal(9.0, 0.5);
                        g.tx(d + (i as i64 % 3), subs[i].1, subs[j].1, a, PaymentType::Wire, "GB", "GB", 0);
                    }
                }
            }
            let a = g.lognormal(8.5, 0.5);
            let i = g.rng.below(subs.len());
            g.tx(d + 3, subs[i].1, ha, a, PaymentType::Wire, "GB", "GB", 0);
        }
        for s in &subs {
            g.normal_activity(s.1, 0, g.days, 0.4, 6.5, "GB");
        }

        // C2 deep holding chain with identified owner, ordinary activity
        let o = g.person(&format!("Chain owner {k}"));
        let mut upper = o;
        for l in 0..(3 + g.rng.below(2)) {
            let e = g.entity(&format!("Chain {k} layer {l} Ltd"), "GB", false);
            g.own(upper, e, 100.0);
            upper = e;
        }
        let a = g.account(upper);
        g.normal_activity(a, 0, g.days, 0.5, 6.8, "GB");

        // C3 owner pays their own company (capital, loans, drawings)
        let (oa, op) = g.fresh_individual();
        let e = g.entity(&format!("Owner {k} Consulting Ltd"), "GB", false);
        g.own(op, e, 100.0);
        let ea = g.account(e);
        for d in (10..g.days).step_by(30) {
            let a = g.lognormal(8.0, 0.4);
            g.tx(d, oa, ea, a, PaymentType::OnlineTransfer, "GB", "GB", 0);
            let a = g.lognormal(7.5, 0.4);
            g.tx(d + 12, ea, oa, a, PaymentType::OnlineTransfer, "GB", "GB", 0);
        }
        g.normal_activity(oa, 0, g.days, 0.3, 6.0, "GB");
        g.normal_activity(ea, 0, g.days, 0.3, 6.5, "GB");

        // C4 opaque but quiet: nominee shares, offshore, unresolved owner; small ordinary payments
        let top = g.entity(&format!("Offshore {k} Trust Co"), "PA", true);
        let e = g.entity(&format!("Opaque {k} Ltd"), "VG", true);
        g.own(top, e, 100.0);
        let a = g.account(e);
        g.normal_activity(a, 0, g.days, 0.25, 5.8, "GB");

        // C5 politically exposed owner, ordinary business
        let mut pp = Party::new(PartyKind::Individual, &format!("PEP owner {k}"));
        pp.pep = PepStatus::Pep;
        pp.countries = vec!["GB".into()];
        let pep = g.c.add_party(pp, &g.ana, t0()).expect("pep");
        let e = g.entity(&format!("PEP {k} Farms Ltd"), "GB", false);
        g.own(pep, e, 100.0);
        let a = g.account(e);
        g.normal_activity(a, 0, g.days, 0.5, 7.0, "GB");

        // C6 near miss: sanctioned person holds 30-45% -> not blocked by the 50% rule, ordinary activity
        let sp = g.confirmed_sanctioned(n_s + k);
        let e = g.entity(&format!("Minority {k} Ltd"), "GB", false);
        let other = g.person(&format!("Co-owner {k}"));
        let pct = 30.0 + 15.0 * g.rng.uniform();
        g.own(sp, e, pct.round());
        g.own(other, e, 100.0 - pct.round());
        let a = g.account(e);
        g.normal_activity(a, 0, g.days, 0.5, 6.8, "GB");

        // C7 namesake false positive: same name as a listed person, different date of birth
        let (sur, given, dob) = Gen::sanctioned_name(n_s + n_c + k);
        let mut p = Party::new(PartyKind::Individual, &format!("{given} {}", capitalise(&sur)));
        p.dob = vec![PartialDate::ymd(dob.year.unwrap_or(1970) + 7, 6, 15)];
        p.nationalities = vec!["GB".into()];
        let ns = g.c.add_party(p, &g.ana, t0()).expect("namesake");
        let _ = g.c.screen_party(ns, &g.ana, t0());
        let ids: Vec<u64> = g.c.alerts().values().filter(|a| a.party == ns && a.status.is_unresolved()).map(|a| a.id).collect();
        for al in ids {
            let _ = g.c.resolve_alert(al, Outcome::FalsePositive, "different date of birth and nationality", &g.mia, t0());
        }
        let e = g.entity(&format!("Namesake {k} Ltd"), "GB", false);
        g.own(ns, e, 100.0);
        let a = g.account(e);
        g.normal_activity(a, 0, g.days, 0.5, 6.8, "GB");

        // C8 family business: four siblings just under the 25% threshold, ordinary activity
        let e = g.entity(&format!("Family {k} Ltd"), "GB", false);
        for j in 0..4 {
            let o = g.person(&format!("Sibling {k}-{j}"));
            g.own(o, e, 24.0);
        }
        let o = g.person(&format!("Minor holder {k}"));
        g.own(o, e, 4.0);
        let a = g.account(e);
        g.normal_activity(a, 0, g.days, 0.5, 6.8, "GB");
    }

    // ---- 8. shell group layering
    for _ in 0..n_s {
        let owner = g.person(&format!("Shell owner {}", g.reg.len()));
        let hc = if g.rng.uniform() < 0.7 { g.hrj() } else { "GB" };
        let nom = g.rng.uniform() < 0.3;
        let hn = format!("Shell Holdings {}", g.reg.len());
        let h = g.entity(&hn, hc, nom);
        g.own(owner, h, 100.0);
        let k = 4 + g.rng.below(3);
        let mut accts = vec![];
        for j in 0..k {
            let sc = if g.rng.uniform() < 0.5 { g.hrj() } else { "GB" };
            let sn = format!("Shell {} {j} Ltd", g.reg.len());
            let e = g.entity(&sn, sc, false);
            g.own(h, e, if j % 2 == 0 { 100.0 } else { 60.0 });
            if j % 2 == 1 {
                g.own(owner, e, 40.0);
            }
            accts.push(g.account(e));
        }
        let (src, _) = g.fresh_individual();
        let (exit, _) = g.fresh_individual();
        let start = g.day(16, 14);
        let total = 40_000.0 + 50_000.0 * g.rng.uniform();
        g.tx(start, src, accts[0], total, PaymentType::Wire, "GB", "GB", 8);
        let mut amt = total;
        let mut d = start + 1;
        for j in 0..k - 1 {
            amt *= 0.94 + 0.05 * g.rng.uniform();
            g.tx(d, accts[j], accts[j + 1], amt, PaymentType::Wire, "GB", "GB", 8);
            d += 1 + g.rng.below(2) as i64;
        }
        let back = accts[k / 2];
        g.tx(d, accts[k - 1], back, amt * 0.3, PaymentType::Wire, "GB", "GB", 8);
        g.tx(d + 1, accts[k - 1], exit, amt * 0.65, PaymentType::Wire, "GB", "GB", 8);
    }

    // ---- 9. sanctioned owner evasion (>= 50% via a nominee vehicle, or direct)
    for i in 0..n_s {
        let s = g.confirmed_sanctioned(i);
        let e = g.entity(&format!("Evasion {i} Trading Ltd"), "GB", false);
        if i % 2 == 0 {
            let vc = g.hrj();
            let v = g.entity(&format!("Evasion {i} Nominees"), vc, true);
            g.own(s, v, 100.0);
            g.own(v, e, 60.0);
            let o = g.person(&format!("Front {i}"));
            g.own(o, e, 40.0);
        } else {
            let pct = 51.0 + (g.rng.uniform() * 30.0).round();
            g.own(s, e, pct);
            let o = g.person(&format!("Front {i}"));
            g.own(o, e, 10.0);
        }
        let a = g.account(e);
        let start = g.day(16, 12);
        let k = 6 + g.rng.below(7);
        let mut total = 0.0;
        for _ in 0..k {
            let src = g.counterparty();
            let amt = 3000.0 + 9000.0 * g.rng.uniform();
            total += amt;
            let d = start + g.rng.below(5) as i64;
            g.tx(d, src, a, amt, PaymentType::OnlineTransfer, "GB", "GB", 9);
        }
        for j in 0..(2 + g.rng.below(2)) {
            let (dst, _) = g.fresh_individual();
            let hc = g.hrj();
            g.tx(start + 6 + j as i64, a, dst, total * 0.3, PaymentType::Wire, "GB", hc, 9);
        }
    }

    // ---- 10. nominee / opaque pass-through (30% look clean: flows are the only signal)
    for i in 0..n_s {
        let clean = g.rng.uniform() < 0.3;
        let e = if clean {
            let o = g.person(&format!("Pass owner {i}"));
            let e = g.entity(&format!("Pass {i} Ltd"), "GB", false);
            g.own(o, e, 100.0);
            e
        } else {
            let (c1, c2) = (g.hrj(), g.hrj());
            let top = g.entity(&format!("Pass {i} Anstalt"), c1, true);
            let e = g.entity(&format!("Pass {i} Ltd"), c2, true);
            g.own(top, e, 100.0);
            e
        };
        let a = g.account(e);
        let start = g.day(16, 10);
        let k = 8 + g.rng.below(8);
        let mut total = 0.0;
        for _ in 0..k {
            let src = if g.rng.uniform() < 0.5 { g.counterparty() } else { g.fresh_individual().0 };
            let amt = 1500.0 + 8000.0 * g.rng.uniform();
            total += amt;
            let d = start + g.rng.below(3) as i64;
            g.tx(d, src, a, amt, PaymentType::OnlineTransfer, "GB", "GB", 10);
        }
        for j in 0..(1 + g.rng.below(3)) {
            let (dst, _) = g.fresh_individual();
            let hc = g.hrj();
            g.tx(start + 2 + 1 + j as i64, a, dst, total * 0.93 / 2.0, PaymentType::Wire, "GB", hc, 10);
        }
    }

    // ---- 11. split-stake structuring through a group of companies with the same four owners
    for i in 0..n_s {
        let owners: Vec<(i64, PartyId)> = (0..4).map(|_| g.fresh_individual()).collect();
        let ents: Vec<(PartyId, i64)> = (0..3)
            .map(|j| {
                let e = g.entity(&format!("Split {i} Co {j}"), "GB", false);
                for (_, p) in &owners {
                    let pct = 22.0 + 2.0 * g.rng.uniform().round();
                    g.own(*p, e, pct);
                }
                (e, g.account(e))
            })
            .collect();
        let (target, _) = g.fresh_individual();
        let start = g.day(16, 14);
        for d in 0..(5 + g.rng.below(4)) as i64 {
            for (oa, _) in &owners {
                if g.rng.uniform() < 0.7 {
                    let e = ents[g.rng.below(3)].1;
                    let amt = 8000.0 + 1900.0 * g.rng.uniform();
                    g.tx(start + d, *oa, e, amt, PaymentType::CashDeposit, "GB", "GB", 11);
                }
            }
        }
        for (_, ea) in &ents {
            let amt = 20_000.0 + 15_000.0 * g.rng.uniform();
            g.tx(start + 10, *ea, target, amt, PaymentType::Wire, "GB", "GB", 11);
        }
    }

    let mut txns = g.out;
    txns.sort_by_key(|t| t.ts);
    for (i, t) in txns.iter_mut().enumerate() {
        t.rf = i as u32;
    }
    OwnershipWorld { txns, compliance: g.c, registry: g.reg, typologies: BO_TYPOLOGIES.iter().map(|s| s.to_string()).collect() }
}
