//! The "up" channel: graph queries and investigation tasks the agent may issue.
//!
//! The grammar is closed. A query is one line, `QUERY <name> key=value …`, where `<name>` is one of a fixed
//! list and every value is a bounded integer or one of a few keywords. Anything else is refused with a reason
//! that is fed back to the agent, never executed. Execution is read-only, goes through [`GraphSource`]
//! (no SQL or free text from the model ever reaches a database) and is bounded in rows, hops and steps.

use crate::evidence::{money, HistoryFacts, OwnershipFacts};
use chrono::DateTime;
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const MAX_ROWS: usize = 2000;
const MAX_STEPS: usize = 4000;

#[derive(Clone, Debug, PartialEq)]
pub struct Flow {
    pub id: i64,
    pub ts: i64,
    pub from: i64,
    pub to: i64,
    pub amount_gbp: f64,
    pub ptype: String,
    pub from_cc: String,
    pub to_cc: String,
}

/// Read-only view of the transaction graph and the registers around it.
pub trait GraphSource {
    /// Flows into or out of `account`, newest first, at most `limit`.
    fn flows_of(&self, account: i64, limit: usize) -> Vec<Flow>;
    fn ownership(&self, _account: i64) -> Option<OwnershipFacts> {
        None
    }
    fn history(&self, _account: i64) -> Option<HistoryFacts> {
        None
    }
}

/// In-memory graph: a pre-fetched neighbourhood (the API builds one per request) or a test fixture.
#[derive(Default, Clone)]
pub struct MemGraph {
    by_account: HashMap<i64, Vec<Flow>>,
    pub ownership: HashMap<i64, OwnershipFacts>,
    pub history: HashMap<i64, HistoryFacts>,
}

impl MemGraph {
    pub fn new(flows: impl IntoIterator<Item = Flow>) -> Self {
        let mut g = MemGraph::default();
        let mut seen = BTreeSet::new();
        for f in flows {
            if !seen.insert(f.id) {
                continue;
            }
            g.by_account.entry(f.from).or_default().push(f.clone());
            g.by_account.entry(f.to).or_default().push(f);
        }
        for v in g.by_account.values_mut() {
            v.sort_by(|a, b| b.ts.cmp(&a.ts).then(b.id.cmp(&a.id)));
        }
        g
    }
    pub fn accounts(&self) -> usize {
        self.by_account.len()
    }
}

impl GraphSource for MemGraph {
    fn flows_of(&self, account: i64, limit: usize) -> Vec<Flow> {
        self.by_account.get(&account).map(|v| v.iter().take(limit.min(MAX_ROWS)).cloned().collect()).unwrap_or_default()
    }
    fn ownership(&self, account: i64) -> Option<OwnershipFacts> {
        self.ownership.get(&account).cloned()
    }
    fn history(&self, account: i64) -> Option<HistoryFacts> {
        self.history.get(&account).cloned()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    In,
    Out,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Query {
    Neighbors { account: i64, depth: u8 },
    Counterparties { account: i64, dir: Dir, top: usize },
    Flows { a: i64, b: i64 },
    Cycles { account: i64, max_len: u8 },
    FanIn { account: i64, days: u32 },
    FanOut { account: i64, days: u32 },
    PassThrough { account: i64 },
    Ownership { account: i64 },
    History { account: i64 },
}

impl Query {
    pub fn render(&self) -> String {
        match self {
            Query::Neighbors { account, depth } => format!("QUERY neighbors account={account} depth={depth}"),
            Query::Counterparties { account, dir, top } => format!("QUERY counterparties account={account} dir={} top={top}", if *dir == Dir::In { "in" } else { "out" }),
            Query::Flows { a, b } => format!("QUERY flows a={a} b={b}"),
            Query::Cycles { account, max_len } => format!("QUERY cycles account={account} max_len={max_len}"),
            Query::FanIn { account, days } => format!("QUERY fan_in account={account} days={days}"),
            Query::FanOut { account, days } => format!("QUERY fan_out account={account} days={days}"),
            Query::PassThrough { account } => format!("QUERY pass_through account={account}"),
            Query::Ownership { account } => format!("QUERY ownership account={account}"),
            Query::History { account } => format!("QUERY history account={account}"),
        }
    }
}

pub const QUERY_NAMES: [&str; 9] = ["neighbors", "counterparties", "flows", "cycles", "fan_in", "fan_out", "pass_through", "ownership", "history"];

fn int(kv: &BTreeMap<String, String>, k: &str, lo: i64, hi: i64, default: Option<i64>) -> Result<i64, String> {
    match kv.get(k) {
        None => default.ok_or_else(|| format!("missing {k}=")),
        Some(v) => {
            if v.is_empty() || v.len() > 18 || !v.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("{k} must be a non-negative integer"));
            }
            let n: i64 = v.parse().map_err(|_| format!("{k} out of range"))?;
            if n < lo || n > hi {
                return Err(format!("{k} must be between {lo} and {hi}"));
            }
            Ok(n)
        }
    }
}

pub fn parse_query(line: &str) -> Result<Query, String> {
    let line = line.trim().trim_start_matches(['-', '*', ' ']).trim();
    let rest = line.strip_prefix("QUERY ").ok_or("a query line starts with 'QUERY '")?;
    let mut parts = rest.split_whitespace();
    let name = parts.next().ok_or("missing query name")?;
    if !QUERY_NAMES.contains(&name) {
        return Err(format!("unknown query '{}'; allowed: {}", name.chars().take(30).collect::<String>(), QUERY_NAMES.join(", ")));
    }
    let mut kv = BTreeMap::new();
    for p in parts {
        let (k, v) = p.split_once('=').ok_or_else(|| format!("'{}' is not key=value", p.chars().take(30).collect::<String>()))?;
        if kv.insert(k.to_string(), v.to_string()).is_some() {
            return Err(format!("duplicate key {k}"));
        }
    }
    let allowed: &[&str] = match name {
        "neighbors" => &["account", "depth"],
        "counterparties" => &["account", "dir", "top"],
        "flows" => &["a", "b"],
        "cycles" => &["account", "max_len"],
        "fan_in" | "fan_out" => &["account", "days"],
        _ => &["account"],
    };
    if let Some(k) = kv.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(format!("{name} does not take '{k}'; it takes {}", allowed.join(", ")));
    }
    const ACC: i64 = 999_999_999_999_999;
    Ok(match name {
        "neighbors" => Query::Neighbors { account: int(&kv, "account", 1, ACC, None)?, depth: int(&kv, "depth", 1, 2, Some(1))? as u8 },
        "counterparties" => {
            let dir = match kv.get("dir").map(String::as_str) {
                Some("in") | None => Dir::In,
                Some("out") => Dir::Out,
                Some(_) => return Err("dir must be in or out".into()),
            };
            Query::Counterparties { account: int(&kv, "account", 1, ACC, None)?, dir, top: int(&kv, "top", 1, 15, Some(5))? as usize }
        }
        "flows" => Query::Flows { a: int(&kv, "a", 1, ACC, None)?, b: int(&kv, "b", 1, ACC, None)? },
        "cycles" => Query::Cycles { account: int(&kv, "account", 1, ACC, None)?, max_len: int(&kv, "max_len", 2, 5, Some(4))? as u8 },
        "fan_in" => Query::FanIn { account: int(&kv, "account", 1, ACC, None)?, days: int(&kv, "days", 1, 365, Some(30))? as u32 },
        "fan_out" => Query::FanOut { account: int(&kv, "account", 1, ACC, None)?, days: int(&kv, "days", 1, 365, Some(30))? as u32 },
        "pass_through" => Query::PassThrough { account: int(&kv, "account", 1, ACC, None)? },
        "ownership" => Query::Ownership { account: int(&kv, "account", 1, ACC, None)? },
        _ => Query::History { account: int(&kv, "account", 1, ACC, None)? },
    })
}

#[derive(Clone, Debug, Default)]
pub struct QueryResult {
    pub query: String,
    pub facts: Vec<String>,
    pub truncated: bool,
}

fn day(ts: i64) -> String {
    DateTime::from_timestamp(ts, 0).map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_else(|| "unknown date".into())
}

fn median(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

pub fn run_query(q: &Query, g: &dyn GraphSource) -> QueryResult {
    let mut r = QueryResult { query: q.render(), ..Default::default() };
    match q {
        Query::Counterparties { account, dir, top } => {
            let fl = g.flows_of(*account, MAX_ROWS);
            r.truncated = fl.len() >= MAX_ROWS;
            let mut m: BTreeMap<i64, (f64, usize, String)> = BTreeMap::new();
            for f in &fl {
                let (other, cc, mine) = match dir {
                    Dir::In if f.to == *account => (f.from, &f.from_cc, true),
                    Dir::Out if f.from == *account => (f.to, &f.to_cc, true),
                    _ => (0, &f.to_cc, false),
                };
                if mine {
                    let e = m.entry(other).or_insert((0.0, 0, cc.clone()));
                    e.0 += f.amount_gbp;
                    e.1 += 1;
                }
            }
            let mut v: Vec<_> = m.into_iter().collect();
            v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
            let word = if *dir == Dir::In { "received from" } else { "sent to" };
            if v.is_empty() {
                r.facts.push(format!("Account {account} has no recorded payments {} in the available data.", if *dir == Dir::In { "received" } else { "sent" }));
            }
            for (acc, (amt, n, cc)) in v.into_iter().take(*top) {
                r.facts.push(format!("Account {account} {word} account {acc} ({cc}): {n} payment(s) totalling {} GBP.", money(amt)));
            }
        }
        Query::Flows { a, b } => {
            let fl: Vec<Flow> = g.flows_of(*a, MAX_ROWS).into_iter().filter(|f| (f.from == *a && f.to == *b) || (f.from == *b && f.to == *a)).collect();
            if fl.is_empty() {
                r.facts.push(format!("No payments between accounts {a} and {b} in the available data."));
            } else {
                let ab: Vec<&Flow> = fl.iter().filter(|f| f.from == *a).collect();
                let ba: Vec<&Flow> = fl.iter().filter(|f| f.from == *b).collect();
                let (lo, hi) = (fl.iter().map(|f| f.ts).min().unwrap(), fl.iter().map(|f| f.ts).max().unwrap());
                r.facts.push(format!(
                    "Between accounts {a} and {b}: {} payment(s) from {a} to {b} totalling {} GBP and {} payment(s) back totalling {} GBP, from {} to {}.",
                    ab.len(), money(ab.iter().map(|f| f.amount_gbp).sum()), ba.len(), money(ba.iter().map(|f| f.amount_gbp).sum()), day(lo), day(hi)
                ));
            }
        }
        Query::FanIn { account, days } | Query::FanOut { account, days } => {
            let inbound = matches!(q, Query::FanIn { .. });
            let fl: Vec<Flow> = g.flows_of(*account, MAX_ROWS).into_iter().filter(|f| if inbound { f.to == *account } else { f.from == *account }).collect();
            r.truncated = fl.len() >= MAX_ROWS;
            let Some(newest) = fl.iter().map(|f| f.ts).max() else {
                r.facts.push(format!("Account {account} has no recorded {} payments in the available data.", if inbound { "inbound" } else { "outbound" }));
                return r;
            };
            let cutoff = newest - *days as i64 * 86_400;
            let w: Vec<&Flow> = fl.iter().filter(|f| f.ts >= cutoff).collect();
            let peers: BTreeSet<i64> = w.iter().map(|f| if inbound { f.from } else { f.to }).collect();
            let mut amts: Vec<f64> = w.iter().map(|f| f.amount_gbp).collect();
            let total: f64 = amts.iter().sum();
            let largest = amts.iter().cloned().fold(0.0, f64::max);
            let med = median(&mut amts);
            let first = w.iter().map(|f| f.ts).min().unwrap_or(newest);
            r.facts.push(format!(
                "In the {days} day(s) to {}, account {account} {} {} payment(s) {} {} distinct account(s), totalling {} GBP (median {} GBP, largest {} GBP, first on {}).",
                day(newest), if inbound { "received" } else { "sent" }, w.len(), if inbound { "from" } else { "to" }, peers.len(), money(total), money(med), money(largest), day(first)
            ));
        }
        Query::PassThrough { account } => {
            let fl = g.flows_of(*account, MAX_ROWS);
            r.truncated = fl.len() >= MAX_ROWS;
            let mut ins: Vec<(i64, f64)> = fl.iter().filter(|f| f.to == *account).map(|f| (f.ts, f.amount_gbp)).collect();
            let mut outs: Vec<(i64, f64)> = fl.iter().filter(|f| f.from == *account).map(|f| (f.ts, f.amount_gbp)).collect();
            ins.sort_by_key(|x| x.0);
            outs.sort_by_key(|x| x.0);
            let total_in: f64 = ins.iter().map(|x| x.1).sum();
            // forward each outflow against inflows received in the previous 48 hours (first-in first-out)
            let mut avail: Vec<(i64, f64)> = ins.clone();
            let mut forwarded = 0.0;
            for (ts, mut amt) in outs.iter().cloned() {
                for slot in avail.iter_mut().filter(|s| s.0 <= ts && ts - s.0 <= 172_800 && s.1 > 0.0) {
                    let take = slot.1.min(amt);
                    slot.1 -= take;
                    amt -= take;
                    forwarded += take;
                    if amt <= 0.0 {
                        break;
                    }
                }
            }
            if total_in <= 0.0 {
                r.facts.push(format!("Account {account} received no payments in the available data, so pass-through cannot be assessed."));
            } else {
                r.facts.push(format!(
                    "Of {} GBP received by account {account}, {} GBP ({:.0}%) was sent onward within two days of arriving.",
                    money(total_in), money(forwarded), 100.0 * forwarded / total_in
                ));
            }
        }
        Query::Neighbors { account, depth } => {
            let fl = g.flows_of(*account, MAX_ROWS);
            r.truncated = fl.len() >= MAX_ROWS;
            let senders: BTreeSet<i64> = fl.iter().filter(|f| f.to == *account).map(|f| f.from).collect();
            let receivers: BTreeSet<i64> = fl.iter().filter(|f| f.from == *account).map(|f| f.to).collect();
            let both: BTreeSet<i64> = senders.intersection(&receivers).cloned().collect();
            r.facts.push(format!(
                "Account {account} exchanged funds with {} distinct account(s): {} sender(s), {} receiver(s), {} both.",
                senders.union(&receivers).count(), senders.len(), receivers.len(), both.len()
            ));
            if *depth >= 2 {
                let mut peers: BTreeMap<i64, f64> = BTreeMap::new();
                for f in &fl {
                    *peers.entry(if f.from == *account { f.to } else { f.from }).or_default() += f.amount_gbp;
                }
                let mut top: Vec<(i64, f64)> = peers.into_iter().collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let first_ring: BTreeSet<i64> = senders.union(&receivers).cloned().collect();
                for (p, _) in top.into_iter().take(5) {
                    let pf = g.flows_of(p, MAX_ROWS);
                    let pp: BTreeSet<i64> = pf.iter().map(|f| if f.from == p { f.to } else { f.from }).filter(|x| *x != *account).collect();
                    let shared = pp.intersection(&first_ring).count();
                    r.facts.push(format!("Counterparty account {p} itself deals with {} other account(s), {} of which also deal with account {account}.", pp.len(), shared));
                }
            }
        }
        Query::Cycles { account, max_len } => {
            let mut steps = 0usize;
            let mut found: BTreeSet<Vec<i64>> = BTreeSet::new();
            let mut detail: BTreeMap<Vec<i64>, (i64, i64, f64)> = BTreeMap::new();
            // depth-first over outgoing flows with non-decreasing timestamps: money must be able to travel the loop
            fn dfs(g: &dyn GraphSource, start: i64, path: &mut Vec<i64>, last_ts: i64, first_ts: i64, min_amt: f64, max_len: usize, steps: &mut usize, found: &mut BTreeSet<Vec<i64>>, detail: &mut BTreeMap<Vec<i64>, (i64, i64, f64)>) {
                if *steps >= MAX_STEPS || found.len() >= 10 {
                    return;
                }
                let cur = *path.last().unwrap();
                for f in g.flows_of(cur, 300).into_iter().filter(|f| f.from == cur && f.ts >= last_ts) {
                    *steps += 1;
                    if *steps >= MAX_STEPS {
                        return;
                    }
                    let first = if path.len() == 1 { f.ts } else { first_ts };
                    let m = if path.len() == 1 { f.amount_gbp } else { min_amt.min(f.amount_gbp) };
                    if f.to == start && path.len() >= 2 {
                        let mut c = path.clone();
                        // canonical rotation: start at the smallest account
                        let k = c.iter().enumerate().min_by_key(|x| x.1).map(|x| x.0).unwrap();
                        c.rotate_left(k);
                        detail.entry(c.clone()).or_insert((first, f.ts, m.min(f.amount_gbp)));
                        found.insert(c);
                    } else if !path.contains(&f.to) && path.len() < max_len {
                        path.push(f.to);
                        dfs(g, start, path, f.ts, first, m, max_len, steps, found, detail);
                        path.pop();
                    }
                }
            }
            dfs(g, *account, &mut vec![*account], i64::MIN, 0, f64::MAX, *max_len as usize, &mut steps, &mut found, &mut detail);
            r.truncated = steps >= MAX_STEPS;
            if found.is_empty() {
                r.facts.push(format!("No loop of up to {max_len} hops returning funds to account {account} was found in the available data{}.", if r.truncated { " (search stopped at its step limit)" } else { "" }));
            }
            for c in found.iter().take(5) {
                let (t0, t1, m) = detail[c];
                let route = c.iter().map(|a| a.to_string()).chain(std::iter::once(c[0].to_string())).collect::<Vec<_>>().join(" -> ");
                r.facts.push(format!("Funds can be traced around the loop {route} ({} hops) between {} and {}; the smallest hop was {} GBP.", c.len(), day(t0), day(t1), money(m)));
            }
        }
        Query::Ownership { account } => match g.ownership(*account) {
            Some(o) => {
                let owners = if o.ubos.is_empty() { "no beneficial owner identified".to_string() } else { o.ubos.iter().take(5).map(|u| format!("{} ({:.1}%)", u.name, u.effective_pct)).collect::<Vec<_>>().join("; ") };
                r.facts.push(format!("Holder of account {account} is {}: {} layer(s), opacity {:.2}, owners: {}.", o.holder, o.layers, o.opacity, owners));
                if o.blocked_ownership_pct > 0.0 {
                    r.facts.push(format!("{:.1}% of the holder of account {account} is owned by blocked parties.", o.blocked_ownership_pct));
                }
            }
            None => r.facts.push(format!("No beneficial-ownership record is available for account {account}.")),
        },
        Query::History { account } => match g.history(*account) {
            Some(h) if h.prior_alerts > 0 => r.facts.push(format!(
                "Account {account} has {} earlier reviewed alert(s): {} SAR(s), {} confirmed, {} false positive(s).",
                h.prior_alerts, h.prior_sar, h.prior_confirmed, h.prior_false_positive
            )),
            _ => r.facts.push(format!("Account {account} has no earlier reviewed alerts.")),
        },
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fl(id: i64, ts: i64, from: i64, to: i64, amt: f64) -> Flow {
        Flow { id, ts, from, to, amount_gbp: amt, ptype: "ACH".into(), from_cc: "GB".into(), to_cc: "GB".into() }
    }
    const D: i64 = 86_400;
    const T0: i64 = 1_665_100_800;

    #[test]
    fn parser_accepts_the_grammar_and_nothing_else() {
        assert_eq!(parse_query("QUERY neighbors account=12 depth=2").unwrap(), Query::Neighbors { account: 12, depth: 2 });
        assert_eq!(parse_query("- QUERY cycles account=5").unwrap(), Query::Cycles { account: 5, max_len: 4 });
        assert_eq!(parse_query("QUERY counterparties account=5 dir=out top=3").unwrap(), Query::Counterparties { account: 5, dir: Dir::Out, top: 3 });
        for bad in [
            "QUERY neighbors account=1; DROP TABLE alerts",
            "QUERY neighbors account=1 depth=9",
            "QUERY neighbors account=-1",
            "QUERY neighbors",
            "QUERY shell account=1",
            "QUERY flows a=1",
            "QUERY fan_in account=1 days=0",
            "QUERY fan_in account=1 limit=5",
            "QUERY history account=1 account=2",
            "neighbors account=1",
            "QUERY ownership account=1e3",
            "QUERY cycles account=1 max_len=50",
        ] {
            assert!(parse_query(bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn render_parses_back() {
        for q in [Query::Neighbors { account: 3, depth: 2 }, Query::Flows { a: 1, b: 2 }, Query::FanIn { account: 9, days: 14 }, Query::Counterparties { account: 4, dir: Dir::Out, top: 7 }, Query::History { account: 8 }] {
            assert_eq!(parse_query(&q.render()).unwrap(), q);
        }
    }

    #[test]
    fn fan_in_counts_distinct_senders_in_the_window() {
        let mut flows = vec![];
        for i in 0..8 {
            flows.push(fl(i, T0 + i * 3600, 100 + i, 1, 900.0 + i as f64));
        }
        flows.push(fl(50, T0 - 90 * D, 777, 1, 5.0)); // outside a 7-day window
        let g = MemGraph::new(flows);
        let r = run_query(&Query::FanIn { account: 1, days: 7 }, &g);
        assert!(r.facts[0].contains("8 payment(s) from 8 distinct account(s)"), "{:?}", r.facts);
    }

    #[test]
    fn finds_a_loop_only_when_time_allows_it() {
        let g = MemGraph::new(vec![fl(1, T0, 1, 2, 5000.0), fl(2, T0 + D, 2, 3, 4900.0), fl(3, T0 + 2 * D, 3, 1, 4800.0)]);
        let r = run_query(&Query::Cycles { account: 1, max_len: 4 }, &g);
        assert!(r.facts[0].contains("1 -> 2 -> 3 -> 1"), "{:?}", r.facts);
        assert!(r.facts[0].contains("4,800.00"));
        // the same edges in the wrong order cannot carry money around the loop
        let g2 = MemGraph::new(vec![fl(1, T0 + 2 * D, 1, 2, 5000.0), fl(2, T0 + D, 2, 3, 4900.0), fl(3, T0, 3, 1, 4800.0)]);
        assert!(run_query(&Query::Cycles { account: 1, max_len: 4 }, &g2).facts[0].starts_with("No loop"));
        // a loop longer than max_len is not reported
        assert!(run_query(&Query::Cycles { account: 1, max_len: 2 }, &g).facts[0].starts_with("No loop"));
    }

    #[test]
    fn pass_through_measures_fast_forwarding() {
        let g = MemGraph::new(vec![fl(1, T0, 10, 1, 1000.0), fl(2, T0 + 3600, 1, 11, 950.0), fl(3, T0 + 10 * D, 12, 1, 500.0)]);
        let r = run_query(&Query::PassThrough { account: 1 }, &g);
        assert!(r.facts[0].contains("950.00 GBP (63%)"), "{:?}", r.facts);
    }

    #[test]
    fn counterparties_flows_and_neighbors() {
        let g = MemGraph::new(vec![fl(1, T0, 5, 1, 100.0), fl(2, T0 + 1, 5, 1, 50.0), fl(3, T0 + 2, 6, 1, 70.0), fl(4, T0 + 3, 1, 5, 20.0)]);
        let c = run_query(&Query::Counterparties { account: 1, dir: Dir::In, top: 5 }, &g);
        assert!(c.facts[0].contains("account 5") && c.facts[0].contains("2 payment(s) totalling 150.00"));
        let f = run_query(&Query::Flows { a: 1, b: 5 }, &g);
        assert!(f.facts[0].contains("2 payment(s) from 1 to 5") == false && f.facts[0].contains("1 payment(s) from 1 to 5"), "{:?}", f.facts);
        let n = run_query(&Query::Neighbors { account: 1, depth: 2 }, &g);
        assert!(n.facts[0].contains("2 distinct account(s)") && n.facts.len() >= 2);
        assert!(run_query(&Query::Neighbors { account: 999, depth: 1 }, &g).facts[0].contains("0 distinct"));
    }

    #[test]
    fn missing_registers_say_so() {
        let g = MemGraph::default();
        assert!(run_query(&Query::Ownership { account: 1 }, &g).facts[0].contains("No beneficial-ownership record"));
        assert!(run_query(&Query::History { account: 1 }, &g).facts[0].contains("no earlier reviewed alerts"));
    }

    #[test]
    fn search_is_bounded() {
        // a dense clique: the step limit must stop the loop search
        let mut flows = vec![];
        let mut id = 0;
        for a in 1..=14 {
            for b in 1..=14 {
                if a != b {
                    id += 1;
                    flows.push(fl(id, T0 + id, a, b, 10.0));
                }
            }
        }
        let g = MemGraph::new(flows);
        let r = run_query(&Query::Cycles { account: 1, max_len: 5 }, &g);
        assert!(r.facts.len() <= 5 && !r.facts.is_empty());
    }
}
