//! Synthetic transaction world for tests and for demonstrating the model when no labelled data is at hand.
//!
//! **This is not evidence about real money laundering.** The generator plants seven laundering typologies
//! whose structure unfolds over *days* (so that a single day looks innocuous), surrounded by benign
//! structure that imitates them on purpose: merchant fan-in, payroll fan-out, legitimate pass-through
//! processors, recurring payments and benign loops. Results on it show whether the machinery can learn
//! such structure, not how it performs on a bank's data. [`write_saml_csv`] writes the stream in the SAML-D
//! column layout so the real-data path (`aml_datasets`) can be tested end to end.

use crate::types::GTxn;
use aml_engine::{Country, PaymentType};
use hcrnn::rng::Rng;
use std::io::Write;
use std::path::Path;

pub const TYPOLOGIES: [&str; 8] =
    ["(normal)", "Fan_In_Structuring", "Cycle", "Layered_Chain", "Slow_Drip", "Dormant_Pass_Through", "Bipartite_Mesh", "Behaviour_Ramp"];

#[derive(Clone, Debug)]
pub struct SynthConfig {
    pub days: i64,
    pub individuals: usize,
    pub merchants: usize,
    pub schemes_per_type: usize,
    pub seed: u64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        SynthConfig { days: 150, individuals: 2500, merchants: 60, schemes_per_type: 14, seed: 11 }
    }
}

pub const DAY: i64 = 86_400;
pub const T0: i64 = 1_665_100_800; // 2022-10-07
const COUNTRIES: [&str; 18] = ["GB", "US", "DE", "FR", "IT", "ES", "NL", "CH", "AT", "JP", "IN", "AE", "MX", "TR", "MA", "NG", "AL", "PK"];
const RISKY: [&str; 7] = ["NG", "MA", "AL", "AE", "MX", "TR", "PK"];

struct World {
    rng: Rng,
    out: Vec<GTxn>,
    next_fresh: i64,
}

fn country(c: &str) -> Country {
    Country::parse(c).unwrap()
}

impl World {
    fn pois(&mut self, lambda: f64) -> usize {
        let l = (-lambda).exp();
        let (mut k, mut p) = (0usize, 1.0);
        loop {
            p *= self.rng.uniform();
            if p <= l {
                return k;
            }
            k += 1;
            if k > 50 {
                return k;
            }
        }
    }
    fn lognormal(&mut self, mu: f64, sigma: f64) -> f64 {
        (mu + sigma * self.rng.normal()).exp()
    }
    fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[self.rng.below(v.len())]
    }
    fn fresh(&mut self) -> i64 {
        self.next_fresh += 1;
        self.next_fresh
    }
    fn ptype(&mut self) -> PaymentType {
        let u = self.rng.uniform();
        match u {
            x if x < 0.25 => PaymentType::Ach,
            x if x < 0.55 => PaymentType::OnlineTransfer,
            x if x < 0.65 => PaymentType::Wire,
            x if x < 0.80 => PaymentType::CreditCard,
            x if x < 0.85 => PaymentType::Cheque,
            x if x < 0.93 => PaymentType::CashDeposit,
            _ => PaymentType::CashWithdrawal,
        }
    }
    fn home(&mut self, risky_p: f64) -> Country {
        if self.rng.uniform() < risky_p {
            country(*self.pick(&RISKY))
        } else if self.rng.uniform() < 0.8 {
            country("GB")
        } else {
            country(*self.pick(&COUNTRIES))
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, day: i64, s: i64, r: i64, amount: f64, pt: PaymentType, sc: Country, rc: Country, typ: u8) {
        let sec = (self.rng.uniform() * DAY as f64) as i64;
        let cross = sc != rc && self.rng.uniform() < 0.7;
        self.out.push(GTxn {
            ts: T0 + day * DAY + sec,
            sender: s,
            receiver: r,
            amount: amount.max(1.0),
            ptype: pt,
            sc,
            rc,
            cross_ccy: cross,
            rf: 0,
            label: typ > 0,
            typology: typ,
            rule: 0.0,
            rule_retro: 0.0,
        });
    }
}

pub fn generate(cfg: &SynthConfig) -> Vec<GTxn> {
    let mut w = World { rng: Rng::new(cfg.seed), out: Vec::new(), next_fresh: 30_000_000 };
    let gb = country("GB");
    let indiv: Vec<i64> = (0..cfg.individuals as i64).map(|i| 10_000_000 + i).collect();
    let merch: Vec<i64> = (0..cfg.merchants as i64).map(|i| 20_000_000 + i).collect();
    let indiv_cc: Vec<Country> = (0..indiv.len()).map(|_| w.home(0.04)).collect();
    let merch_cc: Vec<Country> = (0..merch.len()).map(|_| w.home(0.02)).collect();
    // per-individual profile
    let lambda: Vec<f64> = (0..indiv.len()).map(|_| 0.05 + 0.45 * w.rng.uniform()).collect();
    let mu: Vec<f64> = (0..indiv.len()).map(|_| 4.8 + 0.5 * w.rng.normal()).collect();
    let regular: Vec<Vec<usize>> = (0..indiv.len())
        .map(|_| {
            let k = 3 + w.rng.below(6);
            (0..k).map(|_| w.rng.below(indiv.len() + merch.len())).collect()
        })
        .collect();
    let dormant: Vec<bool> = (0..indiv.len()).map(|_| w.rng.uniform() < 0.08).collect();
    let weekly: Vec<Option<(usize, f64)>> = (0..indiv.len())
        .map(|_| if w.rng.uniform() < 0.2 { Some((w.rng.below(indiv.len() + merch.len()), 80.0 + 220.0 * w.rng.uniform())) } else { None })
        .collect();
    let monthly: Vec<Option<(usize, f64)>> = (0..indiv.len())
        .map(|_| if w.rng.uniform() < 0.1 { Some((w.rng.below(merch.len()), 600.0 + 900.0 * w.rng.uniform())) } else { None })
        .collect();
    let acct = |i: usize| -> (i64, Country) {
        if i < indiv.len() { (indiv[i], indiv_cc[i]) } else { (merch[i - indiv.len()], merch_cc[i - indiv.len()]) }
    };
    let employers: Vec<usize> = (0..20).map(|_| w.rng.below(merch.len())).collect();
    let processors: Vec<(i64, i64)> = (0..12).map(|i| (21_000_000 + i, 22_000_000 + i)).collect(); // processor -> settlement bank
    // benign loops of three family/friend accounts (monthly)
    let loops: Vec<[usize; 3]> = (0..6).map(|_| [w.rng.below(indiv.len()), w.rng.below(indiv.len()), w.rng.below(indiv.len())]).collect();

    for day in 0..cfg.days {
        // ordinary activity
        for i in 0..indiv.len() {
            if dormant[i] && day >= 15 {
                continue;
            }
            let n = w.pois(lambda[i]);
            for _ in 0..n {
                let u = w.rng.uniform();
                let to = if u < 0.7 { regular[i][w.rng.below(regular[i].len())] } else if u < 0.9 { indiv.len() + w.rng.below(merch.len()) } else { w.rng.below(indiv.len()) };
                let (ra, rc) = acct(to);
                if ra == indiv[i] {
                    continue;
                }
                let mut amt = w.lognormal(mu[i], 0.9);
                if w.rng.uniform() < 1.0 / 1500.0 {
                    amt *= 30.0;
                }
                let pt = w.ptype();
                w.push(day, indiv[i], ra, amt, pt, indiv_cc[i], rc, 0);
            }
            if let Some((to, amt)) = weekly[i] {
                if day % 7 == (i as i64) % 7 {
                    let (ra, rc) = acct(to);
                    if ra != indiv[i] {
                        let a = amt * (0.97 + 0.06 * w.rng.uniform());
                        w.push(day, indiv[i], ra, a, PaymentType::Ach, indiv_cc[i], rc, 0);
                    }
                }
            }
            if let Some((to, amt)) = monthly[i] {
                if day % 30 == (i as i64) % 30 {
                    w.push(day, indiv[i], merch[to], amt, PaymentType::Ach, indiv_cc[i], merch_cc[to], 0);
                }
            }
        }
        // merchants pay suppliers weekly (fan-out) and receive from customers via the regular lists above
        for (mi, &m) in merch.iter().enumerate() {
            if day % 7 == (mi as i64) % 7 {
                let k = 5 + w.rng.below(6);
                for _ in 0..k {
                    let to = w.rng.below(merch.len());
                    if to == mi {
                        continue;
                    }
                    let amt = w.lognormal(7.5, 0.6);
                    w.push(day, m, merch[to], amt, PaymentType::Wire, merch_cc[mi], merch_cc[to], 0);
                }
            }
        }
        // payroll fan-out on day 28 of every 30
        if day % 30 == 27 {
            for &e in &employers {
                let k = 15 + w.rng.below(25);
                for _ in 0..k {
                    let to = w.rng.below(indiv.len());
                    if dormant[to] && day >= 15 {
                        continue;
                    }
                    let amt = 2000.0 + 1500.0 * w.rng.uniform();
                    w.push(day, merch[e], indiv[to], amt, PaymentType::Ach, merch_cc[e], indiv_cc[to], 0);
                }
            }
        }
        // legitimate processors: many small payments in, settlement out the next day (benign fan-in + pass-through)
        for (pi, &(p, bank)) in processors.iter().enumerate() {
            let k = 8 + w.rng.below(10);
            let mut total = 0.0;
            for _ in 0..k {
                let from = w.rng.below(indiv.len());
                if dormant[from] && day >= 15 {
                    continue;
                }
                let amt = w.lognormal(4.5, 0.7);
                total += amt;
                w.push(day, indiv[from], p, amt, PaymentType::CreditCard, indiv_cc[from], gb, 0);
            }
            if day % 2 == (pi as i64) % 2 && total > 0.0 {
                w.push(day, p, bank, total * 0.97, PaymentType::Wire, gb, gb, 0);
            }
        }
        // benign monthly loops
        if day % 30 == 12 {
            for l in &loops {
                for k in 0..3 {
                    let (a, b) = (l[k], l[(k + 1) % 3]);
                    if a != b {
                        let amt = 150.0 + 100.0 * w.rng.uniform();
                        w.push(day + k as i64, indiv[a], indiv[b], amt, PaymentType::OnlineTransfer, indiv_cc[a], indiv_cc[b], 0);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ laundering schemes
    let n_s = cfg.schemes_per_type;
    let span = |w: &mut World, lo: i64, extra: i64| -> i64 { lo + (w.rng.uniform() * (cfg.days - lo - extra).max(1) as f64) as i64 };
    let existing = |w: &mut World, indiv: &Vec<i64>, cc: &Vec<Country>| -> (i64, Country) {
        let i = w.rng.below(indiv.len());
        (indiv[i], cc[i])
    };

    // 1. fan-in structuring over several days, then forwarded
    for _ in 0..n_s {
        let start = span(&mut w, 16, 16);
        let mule = w.fresh();
        let mule_cc = w.home(0.3);
        let collector = w.fresh();
        let coll_cc = w.home(0.5);
        let k = 14 + w.rng.below(11);
        let len = 4 + w.rng.below(6) as i64;
        let mut total = 0.0;
        for _ in 0..k {
            let (s, scc) = if w.rng.uniform() < 0.6 { (w.fresh(), w.home(0.2)) } else { existing(&mut w, &indiv, &indiv_cc) };
            let amt = 2000.0 + 7500.0 * w.rng.uniform();
            total += amt;
            let d = start + (w.rng.uniform() * len as f64) as i64;
            let pt = if w.rng.uniform() < 0.5 { PaymentType::CashDeposit } else { PaymentType::OnlineTransfer };
            w.push(d, s, mule, amt, pt, scc, mule_cc, 1);
        }
        let parts = 2 + w.rng.below(3);
        for _ in 0..parts {
            let share = total * (0.88 + 0.08 * w.rng.uniform()) / parts as f64;
            let d = start + len + 1 + w.rng.below(2) as i64;
            w.push(d, mule, collector, share, PaymentType::Wire, mule_cc, coll_cc, 1);
        }
    }
    // 2. cycles (round-tripping) with hops on consecutive days
    for _ in 0..n_s {
        let start = span(&mut w, 16, 12);
        let hops = 3 + w.rng.below(3);
        let members: Vec<(i64, Country)> = (0..hops).map(|_| if w.rng.uniform() < 0.5 { existing(&mut w, &indiv, &indiv_cc) } else { (w.fresh(), w.home(0.2)) }).collect();
        let mut amt = 15_000.0 + 65_000.0 * w.rng.uniform();
        for h in 0..hops {
            let (a, b) = (members[h], members[(h + 1) % hops]);
            if a.0 == b.0 {
                continue;
            }
            let d = start + h as i64 * (1 + w.rng.below(2) as i64);
            w.push(d, a.0, b.0, amt, PaymentType::Wire, a.1, b.1, 2);
            amt *= 0.97;
        }
    }
    // 3. layered chains, one hop every 1-2 days
    for _ in 0..n_s {
        let start = span(&mut w, 16, 14);
        let len = 5 + w.rng.below(3);
        let members: Vec<(i64, Country)> = (0..=len).map(|_| if w.rng.uniform() < 0.4 { existing(&mut w, &indiv, &indiv_cc) } else { (w.fresh(), w.home(0.3)) }).collect();
        let mut amt = 30_000.0 + 60_000.0 * w.rng.uniform();
        let mut d = start;
        for h in 0..len {
            if members[h].0 != members[h + 1].0 {
                w.push(d, members[h].0, members[h + 1].0, amt, PaymentType::Wire, members[h].1, members[h + 1].1, 3);
            }
            amt *= 0.93 + 0.05 * w.rng.uniform();
            d += 1 + w.rng.below(2) as i64;
        }
    }
    // 4. slow drip: one payment a day, a little below 1k, for weeks
    for _ in 0..n_s {
        let start = span(&mut w, 16, 30);
        let (a, acc) = if w.rng.uniform() < 0.5 { existing(&mut w, &indiv, &indiv_cc) } else { (w.fresh(), w.home(0.2)) };
        let (b, bcc) = (w.fresh(), w.home(0.4));
        let len = 14 + w.rng.below(11) as i64;
        for d in 0..len {
            let amt = 700.0 + 280.0 * w.rng.uniform();
            let pt = if w.rng.uniform() < 0.5 { PaymentType::OnlineTransfer } else { PaymentType::Ach };
            w.push(start + d, a, b, amt, pt, acc, bcc, 4);
        }
    }
    // 5. dormant account reactivated and drained within a day
    let eligible: Vec<usize> = (0..indiv.len()).filter(|&i| dormant[i]).collect();
    if !eligible.is_empty() && cfg.days > 100 {
        for s in 0..n_s {
            let d_i = eligible[s % eligible.len()];
            let day = 80 + (w.rng.uniform() * (cfg.days - 86).max(1) as f64) as i64;
            let inbound = 25_000.0 + 35_000.0 * w.rng.uniform();
            let (src, scc) = (w.fresh(), w.home(0.5));
            w.push(day, src, indiv[d_i], inbound, PaymentType::Wire, scc, indiv_cc[d_i], 5);
            let k = 2 + w.rng.below(2);
            for _ in 0..k {
                let (dst, dcc) = (w.fresh(), w.home(0.4));
                let d = day + w.rng.below(2) as i64;
                let amt = inbound * (0.92 + 0.05 * w.rng.uniform()) / k as f64;
                w.push(d, indiv[d_i], dst, amt, PaymentType::Wire, indiv_cc[d_i], dcc, 5);
            }
        }
    }
    // 6. bipartite mesh: 5 senders x 5 receivers, repeated payments over ten days
    for _ in 0..n_s {
        let start = span(&mut w, 16, 14);
        let senders: Vec<(i64, Country)> = (0..5).map(|_| (w.fresh(), w.home(0.2))).collect();
        let recvs: Vec<(i64, Country)> = (0..5).map(|_| (w.fresh(), w.home(0.3))).collect();
        for s in &senders {
            for r in &recvs {
                let times = 3 + w.rng.below(3);
                for _ in 0..times {
                    let d = start + w.rng.below(10) as i64;
                    let amt = 2500.0 + 6500.0 * w.rng.uniform();
                    w.push(d, s.0, r.0, amt, PaymentType::OnlineTransfer, s.1, r.1, 6);
                }
            }
        }
    }
    // 7. behaviour ramp: an established account escalates against two new counterparties
    for _ in 0..n_s {
        let start = span(&mut w, 30, 16);
        let i = w.rng.below(indiv.len());
        if dormant[i] {
            continue;
        }
        let (c1, c2) = ((w.fresh(), w.home(0.3)), (w.fresh(), w.home(0.3)));
        let base = mu[i].exp();
        for d in 0..12i64 {
            let mult = 1.0 + 11.0 * (d as f64 / 11.0).powf(1.5);
            let c = if d % 2 == 0 { c1 } else { c2 };
            let lab = if mult > 3.0 { 7 } else { 0 };
            let amt = base * mult * (0.9 + 0.2 * w.rng.uniform());
            w.push(start + d, indiv[i], c.0, amt, PaymentType::OnlineTransfer, indiv_cc[i], c.1, lab);
        }
    }

    // Decoys: every scheme-only ("fresh") account also does ordinary things throughout the period, so
    // "this account has no other life" is not a free signal.
    let last_fresh = w.next_fresh;
    for f in 30_000_001..=last_fresh {
        let cc = w.home(0.1);
        let k = 6 + w.pois(8.0);
        for _ in 0..k {
            let day = (w.rng.uniform() * cfg.days as f64) as i64;
            let to = w.rng.below(indiv.len() + merch.len());
            let (ra, rc) = acct(to);
            let amt = w.lognormal(4.8, 0.9);
            let pt = w.ptype();
            if w.rng.uniform() < 0.5 {
                w.push(day, f, ra, amt, pt, cc, rc, 0);
            } else {
                w.push(day, ra, f, amt, pt, rc, cc, 0);
            }
        }
    }

    let mut out = w.out;
    out.sort_by_key(|t| t.ts);
    for (i, t) in out.iter_mut().enumerate() {
        t.rf = i as u32;
    }
    out
}

fn country_name(c: Country) -> &'static str {
    match c.as_str() {
        "GB" => "UK", "US" => "USA", "DE" => "Germany", "FR" => "France", "IT" => "Italy", "ES" => "Spain",
        "NL" => "Netherlands", "CH" => "Switzerland", "AT" => "Austria", "JP" => "Japan", "IN" => "India",
        "AE" => "UAE", "MX" => "Mexico", "TR" => "Turkey", "MA" => "Morocco", "NG" => "Nigeria",
        "AL" => "Albania", "PK" => "Pakistan",
        _ => "UK",
    }
}

fn currency_name(c: Country) -> &'static str {
    match c.as_str() {
        "GB" => "UK pounds", "US" => "US dollar", "DE" | "FR" | "IT" | "ES" | "NL" | "AT" => "Euro", "CH" => "Swiss franc",
        "JP" => "Yen", "IN" => "Indian rupee", "AE" => "Dirham", "MX" => "Mexican Peso", "TR" => "Turkish lira",
        "MA" => "Moroccan dirham", "NG" => "Naira", "AL" => "Albanian lek", "PK" => "Pakistani rupee",
        _ => "UK pounds",
    }
}

fn payment_type_name(p: PaymentType) -> &'static str {
    p.as_str()
}

/// Write the stream in the SAML-D column layout (`Time,Date,Sender_account,Receiver_account,Amount,
/// Payment_currency,Received_currency,Sender_bank_location,Receiver_bank_location,Payment_type,
/// Is_laundering,Laundering_type`).
pub fn write_saml_csv(txns: &[GTxn], path: &Path) -> std::io::Result<()> {
    write_saml_csv_named(txns, path, &TYPOLOGIES)
}

/// Like [`write_saml_csv`] with a caller-supplied typology name table (index = `GTxn::typology`).
pub fn write_saml_csv_named<S: AsRef<str>>(txns: &[GTxn], path: &Path, names: &[S]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "Time,Date,Sender_account,Receiver_account,Amount,Payment_currency,Received_currency,Sender_bank_location,Receiver_bank_location,Payment_type,Is_laundering,Laundering_type")?;
    for t in txns {
        let days = t.ts.div_euclid(DAY);
        let sod = t.ts.rem_euclid(DAY);
        // inverse of days_from_civil
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        let pcur = currency_name(t.sc);
        let rcur = if t.cross_ccy {
            let r = currency_name(t.rc);
            if r == pcur { "US dollar" } else { r }
        } else {
            pcur
        };
        writeln!(
            f,
            "{:02}:{:02}:{:02},{:04}-{:02}-{:02},{},{},{:.2},{},{},{},{},{},{},{}",
            sod / 3600, (sod % 3600) / 60, sod % 60, y, m, d, t.sender, t.receiver, t.amount, pcur, rcur,
            country_name(t.sc), country_name(t.rc), payment_type_name(t.ptype), t.label as u8,
            names.get(t.typology as usize).map(|s| s.as_ref()).unwrap_or("(unknown)")
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_deterministic_sorted_and_labelled() {
        let cfg = SynthConfig { days: 110, individuals: 300, merchants: 12, schemes_per_type: 3, seed: 5 };
        let a = generate(&cfg);
        let b = generate(&cfg);
        assert_eq!(a.len(), b.len());
        assert!(a.windows(2).all(|w| w[0].ts <= w[1].ts));
        assert!(a.iter().zip(&b).all(|(x, y)| x.sender == y.sender && x.amount == y.amount && x.ts == y.ts));
        let pos = a.iter().filter(|t| t.label).count();
        assert!(pos > 50 && pos < a.len() / 5, "positives {pos} of {}", a.len());
        for typ in 1..8u8 {
            assert!(a.iter().any(|t| t.typology == typ), "typology {typ} missing");
        }
    }

    #[test]
    fn csv_roundtrips_through_the_saml_adapter() {
        let cfg = SynthConfig { days: 40, individuals: 100, merchants: 8, schemes_per_type: 1, seed: 3 };
        let a = generate(&cfg);
        let dir = std::env::temp_dir().join(format!("aml_graph_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("s.csv");
        write_saml_csv(&a, &p).unwrap();
        let rows: Vec<_> = aml_datasets::open(&p, aml_datasets::Format::SamlD, None)
            .unwrap()
            .filter_map(|i| if let aml_datasets::Item::Row(r) = i { Some(r) } else { None })
            .collect();
        assert_eq!(rows.len(), a.len(), "every row must be mappable");
        for (r, t) in rows.iter().zip(&a) {
            assert_eq!(r.ts, t.ts);
            assert_eq!((r.sender, r.receiver), (t.sender, t.receiver));
            assert_eq!(r.label, Some(t.label));
            assert_eq!(r.scountry, t.sc);
            assert_eq!(r.ptype, t.ptype);
        }
        std::fs::remove_dir_all(dir).ok();
    }
}
