use crate::config::*;
use crate::types::*;
use std::collections::{HashMap, VecDeque};

const DAY: u32 = 86_400;
/// Hard cap on remembered flows per account per direction (bounds memory and scan cost).
const CAP: usize = 256;

#[derive(Clone, Copy)]
struct Flow {
    ts: u32,
    cp: u32,
    amt: f32,
    rf: u32,
    kind: u8,
    depth: u8,
}

#[derive(Default)]
struct Acct {
    first_ts: u32,
    last_ts: u32,
    n_sent: u32,
    n_recv: u32,
    mean: f64,
    m2: f64,
    out: VecDeque<Flow>,
    inn: VecDeque<Flow>,
    fan_in_until: u32,
    fan_out_until: u32,
    hub_until: u32,
    watch: bool,
    /// 0 unknown, 1 low, 2 medium, 3 high
    cust: u8,
    /// declared annual income (GBP), 0 = unknown
    income: f64,
    trade: bool,
    band_until: u32,
}

#[derive(Clone, Copy)]
struct Rule {
    on: bool,
    w: f64,
}

/// Rule parameters compiled out of the (string-keyed) config for hot-loop use.
struct P {
    watch: Rule,
    large: Rule,
    large_thr: f64,
    large_ext: f64,
    large_ext_w: f64,
    hrj: Rule,
    hrj_med_w: f64,
    hrj_vh_w: f64,
    cust_risk: Rule,
    cust_jur: Rule,
    income: Rule,
    inc_single: f64,
    inc_outflow: f64,
    inc_win: u32,
    vhigh: Vec<bool>,
    sanc: Vec<bool>,
    sanctioned: Rule,
    sanc_origin: f64,
    band: Rule,
    band_lo: f64,
    band_hi: f64,
    band_win: u32,
    band_min: usize,
    band_rw: f64,
    band_vs_mean: f64,
    band_hist: u32,
    band_req_hist: bool,
    trade_rule: Rule,
    trade_min: f64,
    fx: Rule,
    fan_in: Rule,
    fan_in_win: u32,
    fan_in_k: usize,
    fan_in_focus: f64,
    fan_out: Rule,
    fan_out_win: u32,
    fan_out_k: usize,
    fan_out_focus: f64,
    hub: Rule,
    hub_win: u32,
    hub_k: usize,
    hub_focus: f64,
    hub_lo: f32,
    hub_hi: f32,
    pt: Rule,
    pt_win: u32,
    pt_min: f64,
    pt_max: f64,
    pt_amt: f64,
    layer: Rule,
    layer_depth: u8,
    cyc: Rule,
    cyc_win: u32,
    cyc_min: usize,
    cyc_max: usize,
    cyc_tol: f64,
    /// > 0: a loop whose closing payment repeats an earlier payment on the same pair (same value within this relative tolerance) is a standing/recurring flow, not new round-tripping
    cyc_repeat: f64,
    large_vs_mean: f64,
    large_hist: u32,
    pair: Rule,
    pair_win: u32,
    pair_n: usize,
    pair_cash_only: bool,
    drop: Rule,
    drop_hist: u32,
    drop_ratio: f64,
    beh: Rule,
    beh_hist: u32,
    beh_z: f64,
    beh_ratio: f64,
    dorm: Rule,
    dorm_secs: u32,
    dorm_amt: f64,
    cash: Rule,
    cash_win: u32,
    cash_n: usize,
    high: Vec<bool>,
    med: Vec<bool>,
    retain: u32,
}

fn compile(cfg: &EngineConfig) -> P {
    let r = |code: &str| {
        let c = cfg.rule(code);
        (Rule { on: c.enabled, w: c.weight.clamp(0.0, 0.99) }, c)
    };
    let days = |c: &RuleCfg, k: &str, d: f64| (c.p(k, d) * DAY as f64) as u32;
    let (watch, _) = r(R_WATCHLIST);
    let (large, lc) = r(R_LARGE);
    let (hrj, hc) = r(R_HRJ);
    let (fx, _) = r(R_FX);
    let (fan_in, fi) = r(R_FAN_IN);
    let (fan_out, fo) = r(R_FAN_OUT);
    let (hub, hu) = r(R_HUB);
    let (pt, pc) = r(R_PASS);
    let (layer, lyc) = r(R_LAYER);
    let (cyc, cy) = r(R_CYCLE);
    let (pair, pa) = r(R_PAIR);
    let (beh, bc) = r(R_BEHAVIOUR);
    let (drop, dpc) = r(R_DROP);
    let (dorm, dc) = r(R_DORMANT);
    let (cash, cac) = r(R_CASH);
    let (cust_risk, _) = r(R_CUST_RISK);
    let (cust_jur, _) = r(R_CUST_JUR);
    let (income, ic) = r(R_INCOME);
    let (sanctioned, snc) = r(R_SANCTIONED);
    let (band, bnd) = r(R_BAND);
    let (trade_rule, trd) = r(R_TRADE);
    let mut sanc = vec![false; 676];
    for c in cfg.sanctioned_countries.iter().filter_map(|s| Country::parse(s)) {
        sanc[c.index()] = true;
    }
    let band_thr = bnd.p("threshold", 10_000.0);
    let mut high = vec![false; 676];
    let mut med = vec![false; 676];
    let mut vhigh = vec![false; 676];
    for c in cfg.very_high_risk_countries.iter().filter_map(|s| Country::parse(s)) {
        vhigh[c.index()] = true;
    }
    for c in cfg.high_risk_countries.iter().filter_map(|s| Country::parse(s)) {
        high[c.index()] = true;
    }
    for c in cfg.medium_risk_countries.iter().filter_map(|s| Country::parse(s)) {
        med[c.index()] = true;
    }
    let p = P {
        watch,
        large,
        large_thr: lc.p("threshold", 100_000.0),
        large_ext: lc.p("extreme", 1_000_000.0),
        large_ext_w: lc.p("extreme_weight", 0.9).clamp(0.0, 0.99),
        hrj,
        hrj_med_w: hc.p("medium_weight", 0.1).clamp(0.0, 0.99),
        hrj_vh_w: hc.p("very_high_weight", 0.4).clamp(0.0, 0.99),
        cust_risk,
        cust_jur,
        income,
        inc_single: ic.p("single_multiple", 0.5),
        inc_outflow: ic.p("outflow_multiple", 1.0),
        inc_win: days(&ic, "window_days", 30.0),
        vhigh,
        sanc,
        sanctioned,
        sanc_origin: snc.p("origin_factor", 0.5).clamp(0.0, 1.0),
        band,
        band_lo: band_thr * (1.0 - bnd.p("band_pct", 0.10)),
        band_hi: band_thr,
        band_win: days(&bnd, "repeat_window_days", 7.0),
        band_min: bnd.p("repeat_min", 2.0) as usize,
        band_rw: bnd.p("repeat_weight", 0.6).clamp(0.0, 0.99),
        band_vs_mean: bnd.p("min_vs_mean", 1.5),
        band_hist: bnd.p("min_history", 5.0) as u32,
        band_req_hist: bnd.p("require_history", 0.0) >= 0.5,
        trade_rule,
        trade_min: trd.p("min_amount", 50_000.0),
        fx,
        fan_in,
        fan_in_win: days(&fi, "window_days", 7.0),
        fan_in_k: fi.p("min_counterparties", 6.0) as usize,
        fan_in_focus: fi.p("min_focus", 0.5),
        fan_out,
        fan_out_win: days(&fo, "window_days", 7.0),
        fan_out_k: fo.p("min_counterparties", 6.0) as usize,
        fan_out_focus: fo.p("min_focus", 0.5),
        hub,
        hub_win: days(&hu, "window_days", 180.0),
        hub_k: hu.p("min_counterparties", 12.0) as usize,
        hub_focus: hu.p("min_focus", 0.5),
        hub_lo: hu.p("amount_min", 1000.0) as f32,
        hub_hi: hu.p("amount_max", 10000.0) as f32,
        pt,
        pt_win: (pc.p("window_hours", 48.0) * 3600.0) as u32,
        pt_min: pc.p("min_ratio", 0.75),
        pt_max: pc.p("max_ratio", 1.3),
        pt_amt: pc.p("min_amount", 500.0),
        layer,
        layer_depth: lyc.p("min_depth", 3.0) as u8,
        cyc,
        cyc_win: days(&cy, "window_days", 14.0),
        cyc_tol: cy.p("amount_tol", 0.0),
        cyc_repeat: cy.p("repeat_tol", 0.0),
        large_vs_mean: lc.p("min_vs_mean", 0.0),
        large_hist: lc.p("min_history", 8.0) as u32,
        cyc_min: cy.p("min_hops", 3.0) as usize,
        cyc_max: cy.p("max_hops", 5.0) as usize,
        pair,
        pair_win: days(&pa, "window_days", 180.0),
        pair_n: pa.p("min_count", 8.0) as usize,
        pair_cash_only: pa.p("cash_only", 1.0) >= 0.5,
        beh,
        drop,
        drop_hist: dpc.p("min_history", 8.0) as u32,
        drop_ratio: dpc.p("ratio", 0.1),
        beh_hist: bc.p("min_history", 8.0) as u32,
        beh_z: bc.p("z_score", 4.0),
        beh_ratio: bc.p("ratio", 5.0),
        dorm,
        dorm_secs: days(&dc, "dormant_days", 60.0),
        dorm_amt: dc.p("min_amount", 5000.0),
        cash,
        cash_win: days(&cac, "window_days", 90.0),
        cash_n: cac.p("min_count", 10.0) as usize,
        high,
        med,
        retain: 0,
    };
    let retain = [p.fan_in_win, p.fan_out_win, p.hub_win, p.pt_win, p.cyc_win, p.pair_win, p.cash_win, p.inc_win, p.band_win, 7 * DAY]
        .into_iter()
        .max()
        .unwrap_or(7 * DAY);
    P { retain, ..p }
}

pub struct Engine {
    cfg: EngineConfig,
    p: P,
    ids: HashMap<i64, u32>,
    accts: Vec<Acct>,
    scratch: Vec<u32>,
    /// Build human-readable `detail` strings on hits (turn off for bulk back-testing).
    pub explain: bool,
}

#[derive(Default)]
struct Updates {
    band: Option<u32>,
    fan_in: Option<u32>,
    fan_out: Option<u32>,
    hub: Option<u32>,
}

fn push_hit(ev: &mut Evaluation, rule: &'static str, weight: f64, side: Side, detail: String) {
    ev.hits.push(Hit { rule, weight, side, detail });
}

fn link_window(dq: &VecDeque<Flow>, since: u32, rule: &'static str, out: &mut Vec<Linked>) {
    for f in dq.iter().rev() {
        if f.ts < since {
            break;
        }
        out.push(Linked { rf: f.rf, rule });
    }
}

/// Distinct counterparties in the window (plus `extra`), optionally restricted to an amount band.
fn distinct(dq: &VecDeque<Flow>, since: u32, extra: u32, scratch: &mut Vec<u32>, band: Option<(f32, f32)>) -> usize {
    scratch.clear();
    scratch.push(extra);
    for f in dq.iter().rev() {
        if f.ts < since {
            break;
        }
        if let Some((lo, hi)) = band {
            if f.amt < lo || f.amt > hi {
                continue;
            }
        }
        scratch.push(f.cp);
    }
    scratch.sort_unstable();
    scratch.dedup();
    scratch.len()
}

struct Node {
    a: u32,
    depth: usize,
    parent: usize,
    rf: u32,
}

/// Bounded BFS for a path r -> ... -> s over recent outbound flows, closing the loop s -> r.
/// `tol > 0` additionally requires every hop to carry about the same value as the triggering payment (|hop - amt| <= tol * amt):
/// real round-tripping preserves value, whereas chance loops in a dense graph do not.
fn find_cycle(accts: &[Acct], s: u32, r: u32, since: u32, min_hops: usize, max_hops: usize, amt: f64, tol: f64, links: &mut Vec<u32>) -> Option<usize> {
    let mut nodes = vec![Node { a: r, depth: 0, parent: usize::MAX, rf: 0 }];
    let mut i = 0;
    while i < nodes.len() {
        let (a, depth) = (nodes[i].a, nodes[i].depth);
        for f in accts[a as usize].out.iter().rev() {
            if f.ts < since {
                break;
            }
            if tol > 0.0 && (f.amt as f64 - amt).abs() > tol * amt {
                continue;
            }
            if f.cp == s {
                let total = depth + 2;
                if total >= min_hops && total <= max_hops {
                    links.push(f.rf);
                    let mut j = i;
                    while nodes[j].parent != usize::MAX {
                        links.push(nodes[j].rf);
                        j = nodes[j].parent;
                    }
                    return Some(total);
                }
            } else if depth + 3 <= max_hops && f.cp != r && nodes.len() < 128 && !nodes.iter().any(|n| n.a == f.cp) {
                nodes.push(Node { a: f.cp, depth: depth + 1, parent: i, rf: f.rf });
            }
        }
        i += 1;
    }
    None
}

fn push_flow(dq: &mut VecDeque<Flow>, flow: Flow, now: u32, retain: u32) {
    while let Some(f) = dq.front() {
        if now.saturating_sub(f.ts) > retain {
            dq.pop_front();
        } else {
            break;
        }
    }
    if dq.len() >= CAP {
        dq.pop_front();
    }
    dq.push_back(flow);
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Self {
        let p = compile(&cfg);
        Engine { cfg, p, ids: HashMap::new(), accts: Vec::new(), scratch: Vec::new(), explain: true }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: EngineConfig) {
        self.p = compile(&cfg);
        self.cfg = cfg;
    }

    pub fn account_count(&self) -> usize {
        self.accts.len()
    }

    pub fn alert_threshold(&self) -> u8 {
        self.cfg.alert_threshold
    }

    /// Weight a retro-linked transaction picks up for `rule` (0 when the rule is off).
    pub fn rule_weight(&self, rule: &str) -> f64 {
        match self.cfg.rules.get(rule) {
            Some(r) if r.enabled => r.weight.clamp(0.0, 0.99),
            _ => 0.0,
        }
    }

    fn idx(&mut self, id: i64) -> u32 {
        if let Some(&i) = self.ids.get(&id) {
            return i;
        }
        let i = self.accts.len() as u32;
        self.accts.push(Acct::default());
        self.ids.insert(id, i);
        i
    }

    pub fn set_watch(&mut self, account: i64, on: bool) {
        let i = self.idx(account) as usize;
        self.accts[i].watch = on;
    }

    /// Attach (or update) the KYC profile of the customer behind `account`.
    pub fn set_profile(&mut self, account: i64, profile: &Profile) {
        let i = self.idx(account) as usize;
        let risk = if profile.pep { CustomerRisk::High } else { profile.risk };
        self.accts[i].cust = match risk {
            CustomerRisk::Low => 1,
            CustomerRisk::Medium => 2,
            CustomerRisk::High => 3,
        };
        self.accts[i].income = profile.annual_income_gbp.filter(|v| *v > 0.0).unwrap_or(0.0);
        self.accts[i].trade = profile.trade_exposed;
    }

    /// Evaluate a transaction, then fold it into the rolling state.
    /// `rf` is an opaque caller reference (DB id / row number) echoed back in `linked`.
    pub fn process(&mut self, t: &TxnIn, rf: u32) -> Evaluation {
        self.step(t, rf, true)
    }

    /// Fold a transaction into the state without scoring (used to hydrate from history).
    pub fn ingest(&mut self, t: &TxnIn, rf: u32) {
        self.step(t, rf, false);
    }

    fn step(&mut self, t: &TxnIn, rf: u32, eval: bool) -> Evaluation {
        let now = t.ts.clamp(0, u32::MAX as i64) as u32;
        let s = self.idx(t.sender);
        let r = self.idx(t.receiver);
        let mut ev = Evaluation::default();
        let (pt_match, in_sum, ratio, in_depth, since_pt) = self.passthrough(s, r, now, t.amount_gbp);
        let new_depth = if pt_match { in_depth.saturating_add(1) } else { 0 };
        let mut upd = Updates::default();
        if eval {
            self.evaluate(&mut ev, &mut upd, t, s, r, now, (pt_match, in_sum, ratio, new_depth, since_pt));
        }
        if let Some(u) = upd.fan_in {
            self.accts[r as usize].fan_in_until = u;
        }
        if let Some(u) = upd.fan_out {
            self.accts[s as usize].fan_out_until = u;
        }
        if let Some(u) = upd.band {
            self.accts[s as usize].band_until = u;
        }
        if let Some(u) = upd.hub {
            self.accts[r as usize].hub_until = u;
        }
        self.apply(t, s, r, now, rf, new_depth);
        ev
    }

    /// Did the sender just forward (roughly) what it recently received?
    fn passthrough(&self, s: u32, r: u32, now: u32, amt: f64) -> (bool, f64, f64, u8, u32) {
        let since = now.saturating_sub(self.p.pt_win);
        if !self.p.pt.on && !self.p.layer.on {
            return (false, 0.0, 0.0, 0, since);
        }
        let a = &self.accts[s as usize];
        let mut in_sum = 0.0f64;
        let mut max_depth = 0u8;
        let mut other = false;
        for f in a.inn.iter().rev() {
            if f.ts < since {
                break;
            }
            in_sum += f.amt as f64;
            max_depth = max_depth.max(f.depth);
            if f.cp != r {
                other = true;
            }
        }
        if in_sum < self.p.pt_amt || !other {
            return (false, in_sum, 0.0, 0, since);
        }
        let mut out_sum = amt;
        for f in a.out.iter().rev() {
            if f.ts < since {
                break;
            }
            out_sum += f.amt as f64;
        }
        let ratio = out_sum / in_sum;
        (ratio >= self.p.pt_min && ratio <= self.p.pt_max, in_sum, ratio, max_depth, since)
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate(&mut self, ev: &mut Evaluation, upd: &mut Updates, t: &TxnIn, s: u32, r: u32, now: u32, pt: (bool, f64, f64, u8, u32)) {
        let explain = self.explain;
        let p = &self.p;
        let accts = &self.accts;
        let scratch = &mut self.scratch;
        let amt = t.amount_gbp;
        let amt32 = amt as f32;
        let same = s == r;
        let (sa, ra) = (&accts[s as usize], &accts[r as usize]);
        let d = |f: std::fmt::Arguments| if explain { f.to_string() } else { String::new() };

        if p.watch.on && (sa.watch || ra.watch) {
            let side = if ra.watch { Side::Receiver } else { Side::Sender };
            push_hit(ev, R_WATCHLIST, p.watch.w, side, d(format_args!("counterparty account is on the watchlist")));
        }
        // `min_vs_mean` > 0: a sender whose normal payments are this large (corporate payer) is not alerted for being itself
        let large_unusual = p.large_vs_mean <= 0.0 || sa.n_sent < p.large_hist || amt >= p.large_vs_mean * sa.mean;
        if p.large.on && amt >= p.large_thr && large_unusual {
            let (w, tag) = if amt >= p.large_ext { (p.large_ext_w, "extreme") } else { (p.large.w, "large") };
            push_hit(ev, R_LARGE, w, Side::Sender, d(format_args!("{tag} value: GBP {amt:.2} (threshold {:.0})", p.large_thr)));
        }
        let (sanc_s, sanc_r) = (p.sanc[t.sender_country.index()], p.sanc[t.receiver_country.index()]);
        let sanc = sanc_s || sanc_r;
        let vh = sanc || p.vhigh[t.sender_country.index()] || p.vhigh[t.receiver_country.index()];
        if p.sanctioned.on && sanc {
            // the counterparty's jurisdiction is the screen; the sender's own country is customer context
            let (side, w) = if sanc_r { (Side::Receiver, p.sanctioned.w) } else { (Side::Sender, p.sanctioned.w * p.sanc_origin) };
            push_hit(ev, R_SANCTIONED, w, side, d(format_args!("sanctioned jurisdiction: {} -> {}", t.sender_country.as_str(), t.receiver_country.as_str())));
        }
        if p.hrj.on && !(p.sanctioned.on && sanc) {
            let hs = p.high[t.sender_country.index()];
            let hr = p.high[t.receiver_country.index()];
            if vh {
                let side = if p.vhigh[t.receiver_country.index()] { Side::Receiver } else { Side::Sender };
                push_hit(ev, R_HRJ, p.hrj_vh_w, side, d(format_args!("very-high-risk jurisdiction: {} -> {}", t.sender_country.as_str(), t.receiver_country.as_str())));
            } else if hs || hr {
                let side = if hr { Side::Receiver } else { Side::Sender };
                push_hit(ev, R_HRJ, p.hrj.w, side, d(format_args!("{} -> {}", t.sender_country.as_str(), t.receiver_country.as_str())));
            } else if p.med[t.sender_country.index()] || p.med[t.receiver_country.index()] {
                push_hit(ev, R_HRJ, p.hrj_med_w, Side::Receiver, d(format_args!("medium-risk jurisdiction")));
            }
        }
        if p.cust_risk.on && (sa.cust == 3 || ra.cust == 3) {
            let side = if sa.cust == 3 { Side::Sender } else { Side::Receiver };
            push_hit(ev, R_CUST_RISK, p.cust_risk.w, side, d(format_args!("high-risk customer on the {} side", if sa.cust == 3 { "sending" } else { "receiving" })));
        }
        if p.cust_jur.on && vh && (sa.cust == 3 || ra.cust == 3) {
            let side = if sa.cust == 3 { Side::Sender } else { Side::Receiver };
            push_hit(ev, R_CUST_JUR, p.cust_jur.w, side, d(format_args!("high-risk customer + very-high-risk jurisdiction")));
        }
        if p.income.on {
            if sa.income > 0.0 {
                let since = now.saturating_sub(p.inc_win);
                let outflow: f64 = amt + sa.out.iter().rev().take_while(|f| f.ts >= since).map(|f| f.amt as f64).sum::<f64>();
                if amt >= p.inc_single * sa.income {
                    push_hit(ev, R_INCOME, p.income.w, Side::Sender, d(format_args!("GBP {amt:.2} is {:.0}% of declared annual income", 100.0 * amt / sa.income)));
                } else if outflow >= p.inc_outflow * sa.income {
                    push_hit(ev, R_INCOME, p.income.w, Side::Sender, d(format_args!("{}-day outflow GBP {outflow:.2} vs declared annual income {:.2}", p.inc_win / DAY, sa.income)));
                }
            } else if ra.income > 0.0 && amt >= p.inc_single * ra.income {
                push_hit(ev, R_INCOME, p.income.w, Side::Receiver, d(format_args!("receipt of GBP {amt:.2} is {:.0}% of receiver's declared annual income", 100.0 * amt / ra.income)));
            }
        }
        if p.fx.on && t.cross_currency {
            push_hit(ev, R_FX, p.fx.w, Side::Sender, d(format_args!("currency conversion")));
        }
        if !same && p.fan_in.on && ra.inn.len() + 1 >= p.fan_in_k {
            let since = now.saturating_sub(p.fan_in_win);
            let n = distinct(&ra.inn, since, s, scratch, None);
            if n >= p.fan_in_k && n as f64 >= p.fan_in_focus * (ra.n_recv as f64 + 1.0) {
                push_hit(ev, R_FAN_IN, p.fan_in.w, Side::Receiver, d(format_args!("{n} distinct senders in {} days", p.fan_in_win / DAY)));
                if ra.fan_in_until < now {
                    link_window(&ra.inn, since, R_FAN_IN, &mut ev.linked);
                    upd.fan_in = Some(now + p.fan_in_win);
                }
            }
        }
        if !same && p.fan_out.on && sa.out.len() + 1 >= p.fan_out_k {
            let since = now.saturating_sub(p.fan_out_win);
            let n = distinct(&sa.out, since, r, scratch, None);
            if n >= p.fan_out_k && n as f64 >= p.fan_out_focus * (sa.n_sent as f64 + 1.0) {
                push_hit(ev, R_FAN_OUT, p.fan_out.w, Side::Sender, d(format_args!("{n} distinct receivers in {} days", p.fan_out_win / DAY)));
                if sa.fan_out_until < now {
                    link_window(&sa.out, since, R_FAN_OUT, &mut ev.linked);
                    upd.fan_out = Some(now + p.fan_out_win);
                }
            }
        }
        if !same && p.hub.on && amt32 >= p.hub_lo && amt32 <= p.hub_hi && ra.inn.len() + 1 >= p.hub_k {
            let since = now.saturating_sub(p.hub_win);
            let n = distinct(&ra.inn, since, s, scratch, Some((p.hub_lo, p.hub_hi)));
            if n >= p.hub_k && n as f64 >= p.hub_focus * (ra.n_recv as f64 + 1.0) {
                push_hit(ev, R_HUB, p.hub.w, Side::Receiver, d(format_args!("{n} distinct senders of sub-threshold amounts in {} days", p.hub_win / DAY)));
                if ra.hub_until < now {
                    link_window_band(&ra.inn, since, (p.hub_lo, p.hub_hi), R_HUB, &mut ev.linked);
                    upd.hub = Some(now + p.hub_win);
                }
            }
        }
        // Just-below-threshold only means something when it is out of pattern for this sender: accounts
        // that routinely move amounts of this size are not structuring.
        let band_unusual = if sa.n_sent >= p.band_hist { amt >= p.band_vs_mean * sa.mean } else { !p.band_req_hist };
        if p.band.on && amt >= p.band_lo && amt < p.band_hi && band_unusual {
            let since = now.saturating_sub(p.band_win);
            let repeats = sa.out.iter().rev().take_while(|f| f.ts >= since).filter(|f| (f.amt as f64) >= p.band_lo && (f.amt as f64) < p.band_hi).count() + 1;
            if repeats >= p.band_min {
                push_hit(ev, R_BAND, p.band_rw.max(p.band.w), Side::Sender, d(format_args!("{repeats} payments just below the {:.0} threshold in {} days", p.band_hi, p.band_win / DAY)));
                if sa.band_until < now {
                    for f in sa.out.iter().rev().take_while(|f| f.ts >= since).filter(|f| (f.amt as f64) >= p.band_lo && (f.amt as f64) < p.band_hi) {
                        ev.linked.push(Linked { rf: f.rf, rule: R_BAND });
                    }
                    upd.band = Some(now + p.band_win);
                }
            } else {
                push_hit(ev, R_BAND, p.band.w, Side::Sender, d(format_args!("GBP {amt:.2} just below the {:.0} reporting threshold", p.band_hi)));
            }
        }
        if p.trade_rule.on && sa.trade && amt >= p.trade_min {
            push_hit(ev, R_TRADE, p.trade_rule.w, Side::Sender, d(format_args!("trade-based customer sending GBP {amt:.2} (min {:.0})", p.trade_min)));
        }
        let (pt_match, in_sum, ratio, new_depth, since_pt) = pt;
        if pt_match {
            if p.pt.on {
                push_hit(ev, R_PASS, p.pt.w, Side::Sender, d(format_args!("forwarded {:.0}% of GBP {in_sum:.2} received in the last {}h", ratio * 100.0, p.pt_win / 3600)));
                link_window(&sa.inn, since_pt, R_PASS, &mut ev.linked);
            }
            if p.layer.on && new_depth >= p.layer_depth {
                push_hit(ev, R_LAYER, p.layer.w, Side::Sender, d(format_args!("pass-through chain depth {new_depth}")));
            }
        }
        let recurring = p.cyc_repeat > 0.0
            && sa.out.iter().rev().any(|f| f.cp == r && (f.amt as f64 - amt).abs() <= p.cyc_repeat * amt);
        if !same && p.cyc.on && !recurring && !ra.out.is_empty() {
            let since = now.saturating_sub(p.cyc_win);
            let mut refs = Vec::new();
            if let Some(h) = find_cycle(accts, s, r, since, p.cyc_min, p.cyc_max, t.amount_gbp, p.cyc_tol, &mut refs) {
                push_hit(ev, R_CYCLE, p.cyc.w, Side::Sender, d(format_args!("funds return to the originator via a {h}-hop loop")));
                ev.linked.extend(refs.into_iter().map(|rf| Linked { rf, rule: R_CYCLE }));
            }
        }
        if !same && p.pair.on && (!p.pair_cash_only || t.payment_type.is_cash()) && sa.out.len() + 1 >= p.pair_n {
            let since = now.saturating_sub(p.pair_win);
            let c = sa.out.iter().rev().take_while(|f| f.ts >= since).filter(|f| f.cp == r).count() + 1;
            if c >= p.pair_n {
                push_hit(ev, R_PAIR, p.pair.w, Side::Sender, d(format_args!("{c} transfers to the same receiver in {} days", p.pair_win / DAY)));
            }
        }
        if p.cash.on && t.payment_type.is_cash() {
            let since = now.saturating_sub(p.cash_win);
            let cnt = |a: &Acct| {
                a.out.iter().rev().take_while(|f| f.ts >= since).filter(|f| f.kind <= 1).count()
                    + a.inn.iter().rev().take_while(|f| f.ts >= since).filter(|f| f.kind <= 1).count()
            };
            let (cs, cr) = (cnt(sa) + 1, cnt(ra) + 1);
            if cs.max(cr) >= p.cash_n {
                let side = if cr > cs { Side::Receiver } else { Side::Sender };
                push_hit(ev, R_CASH, p.cash.w, side, d(format_args!("{} cash movements in {} days", cs.max(cr), p.cash_win / DAY)));
            }
        }
        if p.beh.on && sa.n_sent >= p.beh_hist && sa.mean > 0.0 {
            let var = sa.m2 / (sa.n_sent.saturating_sub(1).max(1)) as f64;
            let sd = var.sqrt().max(0.1 * sa.mean);
            let z = (amt - sa.mean) / sd;
            if z >= p.beh_z || amt >= p.beh_ratio * sa.mean {
                push_hit(ev, R_BEHAVIOUR, p.beh.w, Side::Sender, d(format_args!("GBP {amt:.2} vs usual {:.2} (z={z:.1})", sa.mean)));
            }
        }
        if p.drop.on && sa.n_sent >= p.drop_hist && sa.mean > 0.0 && amt <= p.drop_ratio * sa.mean {
            push_hit(ev, R_DROP, p.drop.w, Side::Sender, d(format_args!("GBP {amt:.2} is {:.1}% of the sender's usual {:.2}", 100.0 * amt / sa.mean, sa.mean)));
        }
        if p.dorm.on && sa.n_sent + sa.n_recv > 0 && now.saturating_sub(sa.last_ts) >= p.dorm_secs && amt >= p.dorm_amt {
            push_hit(ev, R_DORMANT, p.dorm.w, Side::Sender, d(format_args!("dormant for {} days", now.saturating_sub(sa.last_ts) / DAY)));
        }

        let keep: f64 = ev.hits.iter().map(|h| 1.0 - h.weight).product();
        ev.score = ((1.0 - keep) * 100.0).round().clamp(0.0, 100.0) as u8;
    }

    fn apply(&mut self, t: &TxnIn, s: u32, r: u32, now: u32, rf: u32, depth: u8) {
        let kind = t.payment_type as u8;
        let amt32 = t.amount_gbp as f32;
        let retain = self.p.retain;
        {
            let a = &mut self.accts[s as usize];
            if a.n_sent + a.n_recv == 0 {
                a.first_ts = now;
            }
            a.n_sent += 1;
            let x = t.amount_gbp;
            let delta = x - a.mean;
            a.mean += delta / a.n_sent as f64;
            a.m2 += delta * (x - a.mean);
            a.last_ts = a.last_ts.max(now);
            push_flow(&mut a.out, Flow { ts: now, cp: r, amt: amt32, rf, kind, depth }, now, retain);
        }
        {
            let b = &mut self.accts[r as usize];
            if b.n_sent + b.n_recv == 0 {
                b.first_ts = now;
            }
            b.n_recv += 1;
            b.last_ts = b.last_ts.max(now);
            push_flow(&mut b.inn, Flow { ts: now, cp: s, amt: amt32, rf, kind, depth }, now, retain);
        }
    }
}

fn link_window_band(dq: &VecDeque<Flow>, since: u32, band: (f32, f32), rule: &'static str, out: &mut Vec<Linked>) {
    for f in dq.iter().rev() {
        if f.ts < since {
            break;
        }
        if f.amt >= band.0 && f.amt <= band.1 {
            out.push(Linked { rf: f.rf, rule });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tx(ts: i64, s: i64, r: i64, amt: f64) -> TxnIn {
        TxnIn {
            ts,
            sender: s,
            receiver: r,
            amount_gbp: amt,
            payment_type: PaymentType::Ach,
            sender_country: Country::parse("GB").unwrap(),
            receiver_country: Country::parse("GB").unwrap(),
            cross_currency: false,
        }
    }
    fn has(ev: &Evaluation, rule: &str) -> bool {
        ev.hits.iter().any(|h| h.rule == rule)
    }
    fn eng() -> Engine {
        Engine::new(default_config())
    }

    #[test]
    fn fan_in_fires_at_threshold_and_links_earlier_legs() {
        let mut e = eng();
        for i in 0..5 {
            let ev = e.process(&tx(1000 + i, 100 + i, 1, 250.0), i as u32);
            assert!(!has(&ev, R_FAN_IN), "must not fire below threshold");
        }
        let ev = e.process(&tx(1010, 200, 1, 250.0), 5);
        assert!(has(&ev, R_FAN_IN));
        assert_eq!(ev.linked.iter().filter(|l| l.rule == R_FAN_IN).count(), 5);
        // further legs still hit but do not re-link
        let ev = e.process(&tx(1020, 201, 1, 250.0), 6);
        assert!(has(&ev, R_FAN_IN));
        assert!(ev.linked.is_empty());
    }

    #[test]
    fn fan_out_counts_distinct_receivers_only() {
        let mut e = eng();
        for i in 0..10 {
            let ev = e.process(&tx(1000 + i, 1, 50, 100.0), i as u32);
            assert!(!has(&ev, R_FAN_OUT), "same receiver repeatedly is not fan-out");
        }
        let mut e = eng();
        let mut fired = false;
        for i in 0..6 {
            fired = has(&e.process(&tx(1000 + i, 1, 50 + i, 100.0), i as u32), R_FAN_OUT);
        }
        assert!(fired);
    }

    #[test]
    fn passthrough_chain_reaches_layering_depth() {
        let mut e = eng();
        e.process(&tx(0, 1, 2, 1000.0), 0);
        let ev = e.process(&tx(600, 2, 3, 990.0), 1);
        assert!(has(&ev, R_PASS) && !has(&ev, R_LAYER));
        e.process(&tx(1200, 3, 4, 980.0), 2);
        let ev = e.process(&tx(1800, 4, 5, 970.0), 3);
        assert!(has(&ev, R_LAYER), "depth 3 should flag layering");
    }

    #[test]
    fn passthrough_ignores_partial_spend_and_pure_refunds() {
        let mut e = eng();
        e.process(&tx(0, 1, 2, 10_000.0), 0);
        let ev = e.process(&tx(600, 2, 3, 500.0), 1);
        assert!(!has(&ev, R_PASS), "spending 5% of balance is not pass-through");
        let mut e = eng();
        e.process(&tx(0, 1, 2, 1000.0), 0);
        let ev = e.process(&tx(600, 2, 1, 1000.0), 1);
        assert!(!has(&ev, R_PASS), "returning funds to the same sender is a refund, not layering");
    }

    #[test]
    fn three_hop_cycle_detected_but_ping_pong_is_not() {
        let mut e = eng();
        e.process(&tx(0, 1, 2, 5000.0), 0);
        e.process(&tx(100, 2, 3, 5000.0), 1);
        let ev = e.process(&tx(200, 3, 1, 5000.0), 2);
        assert!(has(&ev, R_CYCLE));
        assert_eq!(ev.linked.iter().filter(|l| l.rule == R_CYCLE).count(), 2);
        let mut e = eng();
        e.process(&tx(0, 1, 2, 5000.0), 0);
        let ev = e.process(&tx(100, 2, 1, 5000.0), 1);
        assert!(!has(&ev, R_CYCLE));
    }

    #[test]
    fn cycle_outside_window_is_ignored() {
        let mut e = eng();
        e.process(&tx(0, 1, 2, 5000.0), 0);
        e.process(&tx(100, 2, 3, 5000.0), 1);
        let ev = e.process(&tx(100 + 20 * 86_400, 3, 1, 5000.0), 2);
        assert!(!has(&ev, R_CYCLE));
    }

    #[test]
    fn watchlist_forces_critical_score() {
        let mut e = eng();
        e.set_watch(9, true);
        let ev = e.process(&tx(0, 1, 9, 10.0), 0);
        assert!(has(&ev, R_WATCHLIST));
        assert!(ev.score >= 95);
        assert_eq!(ev.subject_side(), Side::Receiver);
    }

    #[test]
    fn structuring_hub_needs_many_senders_in_band() {
        let mut e = eng();
        let mut last = Evaluation::default();
        for i in 0..12 {
            last = e.process(&tx(i * 86_400 * 5, 1000 + i, 7, 4_900.0), i as u32);
        }
        assert!(has(&last, R_HUB));
        assert_eq!(last.linked.iter().filter(|l| l.rule == R_HUB).count(), 11);
        // same shape but amounts out of band: nothing
        let mut e = eng();
        for i in 0..12 {
            last = e.process(&tx(i * 86_400 * 5, 1000 + i, 7, 40_000.0), i as u32);
        }
        assert!(!has(&last, R_HUB));
    }

    #[test]
    fn jurisdiction_large_and_fx_rules() {
        let mut e = eng();
        let mut t = tx(0, 1, 2, 2_000_000.0);
        t.receiver_country = Country::parse("NG").unwrap();
        t.cross_currency = true;
        let ev = e.process(&t, 0);
        assert!(has(&ev, R_HRJ) && has(&ev, R_FX) && has(&ev, R_LARGE));
        assert!(ev.score >= 90);
    }

    #[test]
    fn amount_drop_is_off_by_default_and_fires_when_enabled() {
        let mk = |on: bool| {
            let mut c = default_config();
            c.rules.get_mut(R_DROP).unwrap().enabled = on;
            Engine::new(c)
        };
        let run = |mut e: Engine| {
            for i in 0..10 {
                e.process(&tx(i * 3600, 1, 100 + i, 500.0 + (i % 3) as f64), i as u32);
            }
            let tiny = e.process(&tx(50_000, 1, 500, 5.0), 99);
            let normal = e.process(&tx(60_000, 1, 501, 480.0), 100);
            (has(&tiny, R_DROP), has(&normal, R_DROP))
        };
        assert_eq!(run(mk(false)), (false, false));
        assert_eq!(run(mk(true)), (true, false));
    }

    #[test]
    fn large_txn_can_ignore_senders_whose_normal_is_large() {
        let run = |vs: f64| {
            let mut c = default_config();
            let l = c.rules.get_mut(R_LARGE).unwrap();
            l.params.insert("min_vs_mean".into(), vs);
            let mut e = Engine::new(c);
            for i in 0..10 {
                e.process(&tx(i * 3600, 1, 100 + i, 400_000.0 + i as f64), i as u32);
            }
            let usual = has(&e.process(&tx(50_000, 1, 500, 420_000.0), 99), R_LARGE);
            let unusual = has(&e.process(&tx(60_000, 1, 501, 5_000_000.0), 100), R_LARGE);
            (usual, unusual)
        };
        assert_eq!(run(0.0), (true, true), "off by default");
        assert_eq!(run(3.0), (false, true));
    }

    #[test]
    fn cycle_amount_tolerance_rejects_chance_loops() {
        let run = |tol: f64, back: f64| {
            let mut c = default_config();
            c.rules.get_mut(R_CYCLE).unwrap().params.insert("amount_tol".into(), tol);
            let mut e = Engine::new(c);
            e.process(&tx(0, 1, 2, 1000.0), 1);
            e.process(&tx(100, 2, 3, 990.0), 2);
            has(&e.process(&tx(200, 3, 1, back), 3), R_CYCLE)
        };
        assert!(run(0.0, 30.0), "no tolerance: any loop counts");
        assert!(run(0.1, 985.0), "value preserved");
        assert!(!run(0.1, 30.0), "value not preserved");
    }

    #[test]
    fn recurring_loop_is_not_new_round_tripping() {
        // the same 3-account ring pays the same amounts every 10 days (standing orders)
        let run = |repeat: f64| {
            let mut c = default_config();
            let r = c.rules.get_mut(R_CYCLE).unwrap();
            r.params.insert("amount_tol".into(), 0.1);
            r.params.insert("repeat_tol".into(), repeat);
            let mut e = Engine::new(c);
            let mut fired = vec![];
            let mut row = 0;
            for round in 0..3i64 {
                let t0 = round * 10 * 86_400;
                e.process(&tx(t0, 1, 2, 1000.0), row);
                e.process(&tx(t0 + 3_600, 2, 3, 990.0), row + 1);
                fired.push(has(&e.process(&tx(t0 + 7_200, 3, 1, 985.0), row + 2), R_CYCLE));
                row += 3;
            }
            fired
        };
        assert_eq!(run(0.0), vec![true, true, true], "off by default: every pass alerts");
        // 3->1 pays 985.0 every round: round 1 is new, rounds 2 and 3 repeat it exactly
        assert_eq!(run(0.0001), vec![true, false, false]);
    }

    #[test]
    fn behaviour_change_vs_own_history() {
        let mut e = eng();
        for i in 0..10 {
            e.process(&tx(i * 3600, 1, 100 + i, 100.0 + (i % 3) as f64), i as u32);
        }
        let ev = e.process(&tx(50_000, 1, 500, 2_000.0), 99);
        assert!(has(&ev, R_BEHAVIOUR));
        let ev = e.process(&tx(60_000, 1, 501, 105.0), 100);
        assert!(!has(&ev, R_BEHAVIOUR));
    }

    #[test]
    fn dormant_account_reactivation() {
        let mut e = eng();
        e.process(&tx(0, 1, 2, 50.0), 0);
        let ev = e.process(&tx(90 * 86_400, 1, 3, 9_000.0), 1);
        assert!(has(&ev, R_DORMANT));
    }

    #[test]
    fn disabled_rule_never_fires_and_config_reload_works() {
        let mut cfg = default_config();
        cfg.rules.get_mut(R_LARGE).unwrap().enabled = false;
        let mut e = Engine::new(cfg);
        assert!(!has(&e.process(&tx(0, 1, 2, 5_000_000.0), 0), R_LARGE));
        let mut cfg = e.config().clone();
        cfg.rules.get_mut(R_LARGE).unwrap().enabled = true;
        e.set_config(cfg);
        assert!(has(&e.process(&tx(1, 1, 2, 5_000_000.0), 1), R_LARGE));
    }

    #[test]
    fn score_is_noisy_or_bounded() {
        let mut e = eng();
        let ev = e.process(&tx(0, 1, 2, 10.0), 0);
        assert_eq!(ev.score, 0);
        e.set_watch(2, true);
        let mut t = tx(1, 1, 2, 5_000_000.0);
        t.receiver_country = Country::parse("NG").unwrap();
        let ev = e.process(&t, 1);
        assert!(ev.score <= 100 && ev.score >= 99);
    }

    #[test]
    fn ingest_updates_state_without_scoring() {
        let mut e = eng();
        for i in 0..5 {
            e.ingest(&tx(1000 + i, 100 + i, 1, 250.0), i as u32);
        }
        let ev = e.process(&tx(1010, 200, 1, 250.0), 5);
        assert!(has(&ev, R_FAN_IN), "hydrated history must count toward windows");
    }

    #[test]
    fn very_high_jurisdiction_outweighs_ordinary_high_risk() {
        let mut e = eng();
        let mut a = tx(0, 1, 2, 100.0);
        a.receiver_country = Country::parse("AF").unwrap();
        let mut b = tx(1, 3, 4, 100.0);
        b.receiver_country = Country::parse("NG").unwrap();
        let (ea, eb) = (e.process(&a, 0), e.process(&b, 1));
        assert!(ea.score > eb.score, "{} vs {}", ea.score, eb.score);
        assert!(ea.score < 60, "jurisdiction alone must not raise an alert at the default threshold");
    }

    #[test]
    fn high_risk_customer_in_very_high_risk_country_alerts() {
        let mut e = eng();
        e.set_profile(1, &Profile { risk: CustomerRisk::High, annual_income_gbp: None, pep: false, trade_exposed: false });
        let mut t = tx(0, 1, 2, 500.0);
        t.receiver_country = Country::parse("AF").unwrap();
        let ev = e.process(&t, 0);
        assert!(has(&ev, R_CUST_RISK) && has(&ev, R_CUST_JUR) && has(&ev, R_HRJ));
        assert!(ev.score >= 60, "combo must alert, got {}", ev.score);
        // same customer, ordinary country: flagged as risky context but no alert
        let ev = e.process(&tx(10, 1, 2, 500.0), 1);
        assert!(has(&ev, R_CUST_RISK) && !has(&ev, R_CUST_JUR) && ev.score < 60);
    }

    #[test]
    fn pep_is_treated_as_high_risk_and_low_rating_is_not() {
        let mut e = eng();
        e.set_profile(1, &Profile { risk: CustomerRisk::Low, annual_income_gbp: None, pep: true, trade_exposed: false });
        e.set_profile(3, &Profile { risk: CustomerRisk::Low, annual_income_gbp: None, pep: false, trade_exposed: false });
        assert!(has(&e.process(&tx(0, 1, 2, 10.0), 0), R_CUST_RISK));
        assert!(!has(&e.process(&tx(1, 3, 4, 10.0), 1), R_CUST_RISK));
    }

    #[test]
    fn income_mismatch_single_and_cumulative_outflow() {
        let mut e = eng();
        e.set_profile(1, &Profile { risk: CustomerRisk::Low, annual_income_gbp: Some(20_000.0), pep: false, trade_exposed: false });
        assert!(!has(&e.process(&tx(0, 1, 2, 5_000.0), 0), R_INCOME));
        assert!(has(&e.process(&tx(10, 1, 3, 12_000.0), 1), R_INCOME), "single >= 50% of annual income");
        let mut e = eng();
        e.set_profile(1, &Profile { risk: CustomerRisk::Low, annual_income_gbp: Some(20_000.0), pep: false, trade_exposed: false });
        for i in 0..4 {
            assert!(!has(&e.process(&tx(i * 3600, 1, 10 + i, 4_900.0), i as u32), R_INCOME));
        }
        assert!(has(&e.process(&tx(5 * 3600, 1, 99, 1_500.0), 9), R_INCOME), "30d outflow >= annual income");
        // no declared income -> rule silent
        let mut e = eng();
        assert!(!has(&e.process(&tx(0, 7, 8, 9_000_000.0), 0), R_INCOME));
    }

    #[test]
    fn new_payment_types_parse() {
        for (s, t) in [("Wire Transfer", PaymentType::Wire), ("Online Transfer", PaymentType::OnlineTransfer),
                       ("Foreign Remittance", PaymentType::CrossBorder), ("Cheque Deposit", PaymentType::Cheque),
                       ("POS", PaymentType::Pos), ("ATM", PaymentType::Atm)] {
            assert_eq!(PaymentType::parse(s), Some(t));
        }
    }

    #[test]
    fn sanctioned_jurisdiction_alerts_alone_and_is_not_double_counted() {
        let mut e = eng();
        let mut t = tx(0, 1, 2, 100.0);
        t.receiver_country = Country::parse("RU").unwrap();
        let ev = e.process(&t, 0);
        assert!(has(&ev, R_SANCTIONED) && !has(&ev, R_HRJ));
        assert!(ev.score >= 60, "sanctioned payment must alert on its own, got {}", ev.score);
        assert_eq!(ev.subject_side(), Side::Receiver);
        // the sender's own country is context: recorded, but not an alert by itself
        let mut t = tx(1, 3, 4, 100.0);
        t.sender_country = Country::parse("KP").unwrap();
        let ev = e.process(&t, 1);
        assert!(has(&ev, R_SANCTIONED) && ev.score < 60 && ev.score > 30, "got {}", ev.score);
        // ordinary high-risk (not sanctioned) stays on the jurisdiction rule
        let mut t = tx(2, 5, 6, 100.0);
        t.receiver_country = Country::parse("NG").unwrap();
        let ev = e.process(&t, 2);
        assert!(!has(&ev, R_SANCTIONED) && has(&ev, R_HRJ));
    }

    fn band_weight(ev: &Evaluation) -> Option<f64> {
        ev.hits.iter().find(|h| h.rule == R_BAND).map(|h| h.weight)
    }

    #[test]
    fn structuring_band_single_then_repeated() {
        let mut e = eng();
        // establish a baseline of small payments so a band payment is out of pattern
        for i in 0..6 {
            e.process(&tx(i * 3600, 1, 50, 900.0), i as u32);
        }
        let t0 = 7 * 3600;
        assert!(band_weight(&e.process(&tx(t0, 1, 2, 8_900.0), 10)).is_none(), "below the band");
        assert!(band_weight(&e.process(&tx(t0 + 1, 1, 2, 10_000.0), 11)).is_none(), "at the threshold is not 'just below'");
        let ev = e.process(&tx(t0 + 100, 1, 3, 9_500.0), 12);
        assert_eq!(band_weight(&ev), Some(0.30), "single band payment is a weak signal");
        let ev = e.process(&tx(t0 + 200, 1, 4, 9_700.0), 13);
        assert_eq!(band_weight(&ev), Some(0.45), "repeat within the window strengthens it, but stays below an alert on its own");
        assert_eq!(ev.linked.iter().filter(|l| l.rule == R_BAND).count(), 1, "earlier band payment is linked");
        // a second band payment weeks later does not count as a repeat
        let mut e = eng();
        for i in 0..6 {
            e.process(&tx(i * 3600, 1, 50, 900.0), i as u32);
        }
        e.process(&tx(7 * 3600, 1, 2, 9_500.0), 10);
        assert_eq!(band_weight(&e.process(&tx(37 * 86_400, 1, 3, 9_500.0), 11)), Some(0.30));
        // an account with no history gives no baseline, so the rule stays quiet by default
        let mut e = eng();
        assert!(band_weight(&e.process(&tx(0, 9, 2, 9_500.0), 0)).is_none());
    }

    #[test]
    fn trade_mispricing_needs_trade_customer_and_size() {
        let mut e = eng();
        e.set_profile(1, &Profile { risk: CustomerRisk::Low, annual_income_gbp: None, pep: false, trade_exposed: true });
        assert!(!has(&e.process(&tx(0, 1, 2, 20_000.0), 0), R_TRADE));
        assert!(has(&e.process(&tx(1, 1, 2, 60_000.0), 1), R_TRADE));
        assert!(!has(&e.process(&tx(2, 5, 6, 600_000.0), 2), R_TRADE), "non-trade customer");
    }

    #[test]
    fn swift_maps_to_cross_border() {
        assert_eq!(PaymentType::parse("SWIFT"), Some(PaymentType::CrossBorder));
    }

    #[test]
    fn structuring_band_ignores_senders_whose_normal_amounts_are_that_large() {
        let mut e = eng();
        for i in 0..8 {
            e.process(&tx(i * 86_400, 1, 100 + i, 9_400.0 + i as f64 * 50.0), i as u32);
        }
        assert!(!has(&e.process(&tx(9 * 86_400, 1, 200, 9_600.0), 20), R_BAND), "9.6k is this sender's norm");
        // same amount from a sender who normally moves small sums is out of pattern
        let mut e = eng();
        for i in 0..8 {
            e.process(&tx(i * 86_400, 2, 100 + i, 800.0 + i as f64 * 10.0), i as u32);
        }
        assert!(has(&e.process(&tx(9 * 86_400, 2, 200, 9_600.0), 20), R_BAND));
    }
}
