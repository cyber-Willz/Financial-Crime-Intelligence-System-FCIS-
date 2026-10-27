//! Score + persist pipeline, config loading and engine hydration.

use crate::{app::AppState, error::ApiError};
use aml_engine::{
    config::ALL_RULES, default_config, Country, CustomerRisk, Engine, EngineConfig, Hit, PaymentType, Profile, Severity,
    Side, TxnIn,
};
use aml_entity::{jurisdiction, rule_config, watchlist};
use chrono::{DateTime, Datelike, Utc};
use futures_util::TryStreamExt;
use sea_orm::{
    sqlx::{self, Row},
    ActiveModelTrait, DatabaseConnection, EntityTrait, Set,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const MAX_BATCH: usize = 5_000;
const ENGINE_CODE: &str = "ENGINE";

#[derive(Debug, Clone, Deserialize)]
pub struct TxnRequest {
    /// Defaults to "now" when omitted.
    pub occurred_at: Option<DateTime<Utc>>,
    pub sender_account: i64,
    pub receiver_account: i64,
    pub amount: f64,
    #[serde(default = "default_currency")]
    pub payment_currency: String,
    /// Defaults to `payment_currency` (no conversion).
    pub received_currency: Option<String>,
    pub sender_country: String,
    pub receiver_country: String,
    /// CASH_DEPOSIT | CASH_WITHDRAWAL | CROSS_BORDER | ACH | CHEQUE | CREDIT_CARD | DEBIT_CARD
    pub payment_type: String,
    /// Pre-normalised GBP amount. When absent the configured FX table is applied.
    pub amount_gbp: Option<f64>,
    pub label_laundering: Option<bool>,
    pub label_typology: Option<String>,
}

fn default_currency() -> String {
    "GBP".into()
}

#[derive(Debug, Serialize)]
pub struct Decision {
    pub transaction_id: i64,
    pub risk_score: u8,
    pub severity: &'static str,
    pub flagged: bool,
    pub hits: Vec<Hit>,
    pub linked_transaction_ids: Vec<u32>,
    pub alert_id: Option<i64>,
}

struct Prepared {
    ts: DateTime<Utc>,
    sender: i64,
    receiver: i64,
    amount: f64,
    gbp: f64,
    pcur: String,
    rcur: String,
    scountry: Country,
    rcountry: Country,
    ptype: PaymentType,
    label: i16,
    typology: String,
    t_in: TxnIn,
}

fn valid_currency(s: &str) -> bool {
    s.len() == 3 && s.bytes().all(|b| b.is_ascii_alphabetic())
}

fn prepare(reqs: Vec<TxnRequest>, cfg: &EngineConfig) -> Result<Vec<Prepared>, ApiError> {
    let now = Utc::now();
    let mut errs = vec![];
    let mut out = Vec::with_capacity(reqs.len());
    for (i, r) in reqs.into_iter().enumerate() {
        let before = errs.len();
        let mut e = |m: &str| errs.push(format!("transactions[{i}]: {m}"));
        if r.sender_account <= 0 || r.receiver_account <= 0 {
            e("account numbers must be positive");
        }
        if r.sender_account == r.receiver_account {
            e("sender and receiver must differ");
        }
        if !(r.amount.is_finite() && r.amount > 0.0 && r.amount < 1e15) {
            e("amount must be a positive, finite number");
        }
        if let Some(g) = r.amount_gbp {
            if !(g.is_finite() && g >= 0.0 && g < 1e15) {
                e("amount_gbp must be a non-negative, finite number");
            }
        }
        let pcur = r.payment_currency.trim().to_ascii_uppercase();
        let rcur = r.received_currency.as_deref().map(|s| s.trim().to_ascii_uppercase()).unwrap_or_else(|| pcur.clone());
        if !valid_currency(&pcur) || !valid_currency(&rcur) {
            e("currencies must be 3-letter ISO-4217 codes");
        }
        let sc = Country::parse(&r.sender_country);
        let rc = Country::parse(&r.receiver_country);
        if sc.is_none() || rc.is_none() {
            e("countries must be 2-letter ISO-3166 codes");
        }
        let pt = PaymentType::parse(&r.payment_type);
        if pt.is_none() {
            e("unknown payment_type");
        }
        if let Some(t) = &r.label_typology {
            if t.len() > 40 {
                e("label_typology too long");
            }
        }
        let ts = r.occurred_at.unwrap_or(now);
        if ts > now + chrono::Duration::days(1) {
            e("occurred_at is in the future");
        }
        if errs.len() > before {
            continue;
        }
        let gbp = r.amount_gbp.unwrap_or_else(|| cfg.to_gbp(r.amount, &pcur));
        let (sc, rc, pt) = (sc.unwrap(), rc.unwrap(), pt.unwrap());
        out.push(Prepared {
            ts,
            sender: r.sender_account,
            receiver: r.receiver_account,
            amount: r.amount,
            gbp,
            t_in: TxnIn {
                ts: ts.timestamp(),
                sender: r.sender_account,
                receiver: r.receiver_account,
                amount_gbp: gbp,
                payment_type: pt,
                sender_country: sc,
                receiver_country: rc,
                cross_currency: pcur != rcur,
            },
            pcur,
            rcur,
            scountry: sc,
            rcountry: rc,
            ptype: pt,
            label: r.label_laundering.map(|b| b as i16).unwrap_or(-1),
            typology: r.label_typology.unwrap_or_default(),
        });
    }
    if errs.is_empty() {
        Ok(out)
    } else {
        errs.truncate(50);
        Err(ApiError::Unprocessable(errs))
    }
}

struct AcctAgg {
    country: String,
    first: DateTime<Utc>,
    last: DateTime<Utc>,
    sent_n: i64,
    recv_n: i64,
    sent: f64,
    recv: f64,
    risk: u8,
    alerts: i32,
}

struct AlertAgg {
    subject: i64,
    first: i64,
    last: i64,
    count: i32,
    score: u8,
    rules: BTreeSet<&'static str>,
    explanation: serde_json::Value,
}

pub fn day_key(ts: DateTime<Utc>) -> String {
    format!("{:04}{:02}{:02}", ts.year(), ts.month(), ts.day())
}

/// Score a batch against the engine and persist transactions, accounts, alerts and
/// retroactive pattern links atomically. Decisions come back in request order.
pub async fn ingest(st: &AppState, reqs: Vec<TxnRequest>) -> Result<Vec<Decision>, ApiError> {
    if reqs.is_empty() {
        return Ok(vec![]);
    }
    if reqs.len() > MAX_BATCH {
        return Err(ApiError::BadRequest(format!("batch too large (max {MAX_BATCH})")));
    }
    let pool = st.db.get_postgres_connection_pool();
    let mut engine = st.engine.lock().await;
    let threshold = engine.alert_threshold();
    let prepared = prepare(reqs, engine.config())?;
    let n = prepared.len();

    // Process chronologically (windows assume time order); report in request order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| prepared[i].ts);

    let ids: Vec<i64> = sqlx::query_scalar("SELECT nextval('transactions_id_seq') FROM generate_series(1, $1::int)")
        .bind(n as i32)
        .fetch_all(pool)
        .await?;
    let mut ids = ids;
    ids.sort_unstable();
    if ids.last().copied().unwrap_or(0) > u32::MAX as i64 {
        return Err(ApiError::Internal(anyhow::anyhow!("transaction id space exhausted for engine references")));
    }

    let mut evals = Vec::with_capacity(n);
    for (k, &i) in order.iter().enumerate() {
        evals.push((i, ids[k], engine.process(&prepared[i].t_in, ids[k] as u32)));
    }

    let result = persist(pool, &engine, &prepared, &evals, threshold).await;
    let alert_ids = match result {
        Ok(m) => m,
        Err(e) => {
            // The engine already folded this batch in; rebuild it from what the DB really holds.
            tracing::error!(error = ?e, "persist failed; rebuilding engine state from database");
            match load_engine(&st.db, st.hydrate_days).await {
                Ok(fresh) => *engine = fresh,
                Err(e2) => tracing::error!(error = ?e2, "engine rebuild failed"),
            }
            return Err(e);
        }
    };

    let mut decisions: Vec<Option<Decision>> = (0..n).map(|_| None).collect();
    for (i, id, ev) in evals {
        let flagged = ev.score >= threshold;
        let key = alert_key(&prepared[i], &ev);
        decisions[i] = Some(Decision {
            transaction_id: id,
            risk_score: ev.score,
            severity: Severity::from_score(ev.score).as_str(),
            flagged,
            linked_transaction_ids: ev.linked.iter().map(|l| l.rf).collect::<BTreeSet<_>>().into_iter().collect(),
            alert_id: if flagged { key.and_then(|k| alert_ids.get(&k).copied()) } else { None },
            hits: ev.hits,
        });
    }
    Ok(decisions.into_iter().map(|d| d.expect("every slot filled")).collect())
}

fn subject_of(p: &Prepared, ev: &aml_engine::Evaluation) -> i64 {
    match ev.subject_side() {
        Side::Receiver => p.receiver,
        Side::Sender => p.sender,
    }
}

fn alert_key(p: &Prepared, ev: &aml_engine::Evaluation) -> Option<String> {
    let top = ev.hits.iter().max_by(|a, b| a.weight.total_cmp(&b.weight))?;
    Some(format!("{}:{}:{}", subject_of(p, ev), day_key(p.ts), top.rule))
}

type Evals = Vec<(usize, i64, aml_engine::Evaluation)>;

async fn persist(
    pool: &sqlx::PgPool,
    engine: &Engine,
    prepared: &[Prepared],
    evals: &Evals,
    threshold: u8,
) -> Result<HashMap<String, i64>, ApiError> {
    let mut accts: BTreeMap<i64, AcctAgg> = BTreeMap::new();
    let mut alerts: BTreeMap<String, AlertAgg> = BTreeMap::new();
    let mut retro: BTreeMap<&'static str, BTreeSet<i64>> = BTreeMap::new();

    for (i, id, ev) in evals {
        let p = &prepared[*i];
        for (acct, country, is_sender) in [(p.sender, p.scountry, true), (p.receiver, p.rcountry, false)] {
            let a = accts.entry(acct).or_insert_with(|| AcctAgg {
                country: country.as_str().to_string(),
                first: p.ts,
                last: p.ts,
                sent_n: 0,
                recv_n: 0,
                sent: 0.0,
                recv: 0.0,
                risk: 0,
                alerts: 0,
            });
            a.first = a.first.min(p.ts);
            a.last = a.last.max(p.ts);
            if is_sender {
                a.sent_n += 1;
                a.sent += p.gbp;
            } else {
                a.recv_n += 1;
                a.recv += p.gbp;
            }
        }
        if ev.score > 0 {
            // the account that *drives* the risk carries it (not the innocent counterparty)
            let subj = subject_of(p, ev);
            let a = accts.get_mut(&subj).expect("subject is sender or receiver");
            a.risk = a.risk.max(ev.score);
        }
        if ev.score >= threshold {
            if let Some(key) = alert_key(p, ev) {
                let subj = subject_of(p, ev);
                accts.get_mut(&subj).unwrap().alerts += 1;
                let top_changes = alerts.get(&key).map(|e| ev.score > e.score).unwrap_or(true);
                let entry = alerts.entry(key).or_insert_with(|| AlertAgg {
                    subject: subj,
                    first: *id,
                    last: *id,
                    count: 0,
                    score: 0,
                    rules: BTreeSet::new(),
                    explanation: serde_json::Value::Null,
                });
                entry.first = entry.first.min(*id);
                entry.last = entry.last.max(*id);
                entry.count += 1;
                entry.rules.extend(ev.hits.iter().map(|h| h.rule));
                if top_changes {
                    entry.score = ev.score;
                    entry.explanation = serde_json::json!({
                        "score": ev.score,
                        "transaction_id": id,
                        "occurred_at": p.ts,
                        "sender_account": p.sender,
                        "receiver_account": p.receiver,
                        "amount_gbp": p.gbp,
                        "payment_type": p.ptype.as_str(),
                        "hits": ev.hits,
                        "linked_transaction_ids": ev.linked.iter().map(|l| l.rf).collect::<BTreeSet<_>>(),
                    });
                }
            }
        }
        for l in &ev.linked {
            retro.entry(l.rule).or_default().insert(l.rf as i64);
        }
    }

    let mut tx = pool.begin().await?;

    // 1. accounts (FK target for transactions + alerts)
    {
        let keys: Vec<i64> = accts.keys().copied().collect();
        let v = |f: &dyn Fn(&AcctAgg) -> f64| accts.values().map(|a| f(a)).collect::<Vec<f64>>();
        sqlx::query(
            "INSERT INTO accounts (account_number, country, first_seen, last_seen, txn_sent_count, txn_received_count, \
                total_sent, total_received, risk_score, risk_level, alert_count) \
             SELECT a, c, fs, ls, sn, rn, ROUND(sv::numeric,2), ROUND(rv::numeric,2), rs, \
                CASE WHEN rs>=90 THEN 'critical' WHEN rs>=75 THEN 'high' WHEN rs>=60 THEN 'medium' ELSE 'low' END, ac \
             FROM UNNEST($1::bigint[], $2::text[], $3::timestamptz[], $4::timestamptz[], $5::bigint[], $6::bigint[], \
                $7::float8[], $8::float8[], $9::smallint[], $10::int[]) AS u(a,c,fs,ls,sn,rn,sv,rv,rs,ac) \
             ON CONFLICT (account_number) DO UPDATE SET \
                country = COALESCE(accounts.country, EXCLUDED.country), \
                first_seen = LEAST(accounts.first_seen, EXCLUDED.first_seen), \
                last_seen = GREATEST(accounts.last_seen, EXCLUDED.last_seen), \
                txn_sent_count = accounts.txn_sent_count + EXCLUDED.txn_sent_count, \
                txn_received_count = accounts.txn_received_count + EXCLUDED.txn_received_count, \
                total_sent = accounts.total_sent + EXCLUDED.total_sent, \
                total_received = accounts.total_received + EXCLUDED.total_received, \
                risk_score = GREATEST(accounts.risk_score, EXCLUDED.risk_score), \
                risk_level = CASE WHEN GREATEST(accounts.risk_score, EXCLUDED.risk_score) >= 90 THEN 'critical' \
                                  WHEN GREATEST(accounts.risk_score, EXCLUDED.risk_score) >= 75 THEN 'high' \
                                  WHEN GREATEST(accounts.risk_score, EXCLUDED.risk_score) >= 60 THEN 'medium' ELSE 'low' END, \
                alert_count = accounts.alert_count + EXCLUDED.alert_count, \
                updated_at = now()",
        )
        .bind(keys)
        .bind(accts.values().map(|a| a.country.clone()).collect::<Vec<_>>())
        .bind(accts.values().map(|a| a.first).collect::<Vec<_>>())
        .bind(accts.values().map(|a| a.last).collect::<Vec<_>>())
        .bind(accts.values().map(|a| a.sent_n).collect::<Vec<_>>())
        .bind(accts.values().map(|a| a.recv_n).collect::<Vec<_>>())
        .bind(v(&|a| a.sent))
        .bind(v(&|a| a.recv))
        .bind(accts.values().map(|a| a.risk as i16).collect::<Vec<_>>())
        .bind(accts.values().map(|a| a.alerts).collect::<Vec<_>>())
        .execute(&mut *tx)
        .await?;
    }

    // 2. transactions
    {
        let m = evals.len();
        let (mut id, mut ts, mut s, mut r, mut amt, mut pc, mut rc, mut gbp) = (
            Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m),
            Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m),
        );
        let (mut sc, mut rcc, mut pt, mut score, mut flagged, mut rules, mut lab, mut typ) = (
            Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m),
            Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m),
        );
        for (i, tid, ev) in evals {
            let p = &prepared[*i];
            id.push(*tid);
            ts.push(p.ts);
            s.push(p.sender);
            r.push(p.receiver);
            amt.push(p.amount);
            pc.push(p.pcur.clone());
            rc.push(p.rcur.clone());
            gbp.push(p.gbp);
            sc.push(p.scountry.as_str().to_string());
            rcc.push(p.rcountry.as_str().to_string());
            pt.push(p.ptype.as_str().to_string());
            score.push(ev.score as i16);
            flagged.push(ev.score >= threshold);
            rules.push(serde_json::to_string(&ev.rule_codes()).unwrap_or_else(|_| "[]".into()));
            lab.push(p.label);
            typ.push(p.typology.clone());
        }
        sqlx::query(
            "INSERT INTO transactions (id, occurred_at, sender_account, receiver_account, amount, payment_currency, \
                received_currency, amount_gbp, sender_country, receiver_country, payment_type, risk_score, flagged, \
                rule_codes, label_laundering, label_typology) \
             SELECT id, ts, s, r, ROUND(amt::numeric,2), pc, rc, ROUND(gbp::numeric,2), sc, rcc, pt, score, fl, \
                rules::jsonb, CASE WHEN lab < 0 THEN NULL ELSE lab = 1 END, NULLIF(typ,'') \
             FROM UNNEST($1::bigint[], $2::timestamptz[], $3::bigint[], $4::bigint[], $5::float8[], $6::text[], \
                $7::text[], $8::float8[], $9::text[], $10::text[], $11::text[], $12::smallint[], $13::bool[], \
                $14::text[], $15::smallint[], $16::text[]) \
                AS u(id,ts,s,r,amt,pc,rc,gbp,sc,rcc,pt,score,fl,rules,lab,typ)",
        )
        .bind(id).bind(ts).bind(s).bind(r).bind(amt).bind(pc).bind(rc).bind(gbp)
        .bind(sc).bind(rcc).bind(pt).bind(score).bind(flagged).bind(rules).bind(lab).bind(typ)
        .execute(&mut *tx)
        .await?;
    }

    // 3. alerts (one per subject/day/primary-rule; repeats fold into the same alert)
    let mut alert_ids = HashMap::new();
    if !alerts.is_empty() {
        let keys: Vec<String> = alerts.keys().cloned().collect();
        let rows = sqlx::query(
            "INSERT INTO alerts (dedupe_key, subject_account, first_txn_id, last_txn_id, txn_count, score, severity, \
                rule_codes, explanation) \
             SELECT k, s, f, l, n, sc, sev, rc::jsonb, ex::jsonb \
             FROM UNNEST($1::text[], $2::bigint[], $3::bigint[], $4::bigint[], $5::int[], $6::smallint[], $7::text[], \
                $8::text[], $9::text[]) AS u(k,s,f,l,n,sc,sev,rc,ex) \
             ON CONFLICT (dedupe_key) DO UPDATE SET \
                first_txn_id = LEAST(alerts.first_txn_id, EXCLUDED.first_txn_id), \
                last_txn_id  = GREATEST(alerts.last_txn_id, EXCLUDED.last_txn_id), \
                txn_count    = alerts.txn_count + EXCLUDED.txn_count, \
                severity     = CASE WHEN EXCLUDED.score > alerts.score THEN EXCLUDED.severity ELSE alerts.severity END, \
                explanation  = CASE WHEN EXCLUDED.score > alerts.score THEN EXCLUDED.explanation ELSE alerts.explanation END, \
                score        = GREATEST(alerts.score, EXCLUDED.score), \
                rule_codes   = (SELECT COALESCE(jsonb_agg(DISTINCT x), '[]'::jsonb) \
                                FROM jsonb_array_elements(alerts.rule_codes || EXCLUDED.rule_codes) AS t(x)), \
                updated_at   = now() \
             RETURNING id, dedupe_key",
        )
        .bind(keys)
        .bind(alerts.values().map(|a| a.subject).collect::<Vec<_>>())
        .bind(alerts.values().map(|a| a.first).collect::<Vec<_>>())
        .bind(alerts.values().map(|a| a.last).collect::<Vec<_>>())
        .bind(alerts.values().map(|a| a.count).collect::<Vec<_>>())
        .bind(alerts.values().map(|a| a.score as i16).collect::<Vec<_>>())
        .bind(alerts.values().map(|a| Severity::from_score(a.score).as_str().to_string()).collect::<Vec<_>>())
        .bind(alerts.values().map(|a| serde_json::to_string(&a.rules).unwrap_or_else(|_| "[]".into())).collect::<Vec<_>>())
        .bind(alerts.values().map(|a| a.explanation.to_string()).collect::<Vec<_>>())
        .fetch_all(&mut *tx)
        .await?;
        for row in rows {
            alert_ids.insert(row.get::<String, _>("dedupe_key"), row.get::<i64, _>("id"));
        }
    }

    // 4. retroactive pattern links: earlier legs of a pattern pick up that rule's weight (idempotent per rule)
    for (rule, ids) in retro {
        let w = engine.rule_weight(rule);
        if w <= 0.0 {
            continue;
        }
        sqlx::query(
            "UPDATE transactions SET \
                risk_score = (LEAST(100, ROUND(100 - (100 - risk_score)::float8 * (1 - $3::float8))))::smallint, \
                flagged = (LEAST(100, ROUND(100 - (100 - risk_score)::float8 * (1 - $3::float8))) >= $4::float8), \
                rule_codes = rule_codes || to_jsonb($2::text) \
             WHERE id = ANY($1::bigint[]) AND NOT (rule_codes ? $2::text)",
        )
        .bind(ids.into_iter().collect::<Vec<_>>())
        .bind(rule)
        .bind(w)
        .bind(threshold as f64)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(alert_ids)
}

// ---------------------------------------------------------------- config & hydration

fn country_name(code: &str) -> &'static str {
    match code {
        "NG" => "Nigeria", "MA" => "Morocco", "AL" => "Albania", "AE" => "United Arab Emirates",
        "MX" => "Mexico", "TR" => "Turkey", "PK" => "Pakistan", "AF" => "Afghanistan",
        "KP" => "North Korea", "MM" => "Myanmar", "IR" => "Iran", "SY" => "Syria", "RU" => "Russia",
        "SD" => "Sudan", "VE" => "Venezuela", _ => "Unknown",
    }
}

/// Load the engine configuration from the database, seeding any missing rows from the defaults.
pub async fn load_config(db: &DatabaseConnection) -> anyhow::Result<EngineConfig> {
    let mut cfg = default_config();
    let existing: HashMap<String, rule_config::Model> =
        rule_config::Entity::find().all(db).await?.into_iter().map(|m| (m.code.clone(), m)).collect();

    for code in ALL_RULES {
        let def = cfg.rules.get(code).cloned().expect("default rule");
        match existing.get(code) {
            None => {
                rule_config::ActiveModel {
                    code: Set(code.into()),
                    description: Set(def.description.clone()),
                    enabled: Set(def.enabled),
                    weight: Set(def.weight),
                    params: Set(serde_json::to_value(&def.params)?),
                    updated_at: Set(Utc::now()),
                    updated_by: Set(Some("seed".into())),
                }
                .insert(db)
                .await?;
            }
            Some(m) => {
                let r = cfg.rules.get_mut(code).unwrap();
                r.enabled = m.enabled;
                r.weight = m.weight;
                if let Ok(p) = serde_json::from_value::<BTreeMap<String, f64>>(m.params.clone()) {
                    for (k, v) in p {
                        r.params.insert(k, v);
                    }
                }
            }
        }
    }
    match existing.get(ENGINE_CODE) {
        None => {
            rule_config::ActiveModel {
                code: Set(ENGINE_CODE.into()),
                description: Set("Global engine settings".into()),
                enabled: Set(true),
                weight: Set(0.0),
                params: Set(serde_json::json!({"alert_threshold": cfg.alert_threshold})),
                updated_at: Set(Utc::now()),
                updated_by: Set(Some("seed".into())),
            }
            .insert(db)
            .await?;
        }
        Some(m) => {
            if let Some(t) = m.params.get("alert_threshold").and_then(|v| v.as_u64()) {
                cfg.alert_threshold = t.clamp(1, 100) as u8;
            }
        }
    }

    // Seed any default jurisdiction that has no row yet (never overrides an admin's edits).
    let mut jur = jurisdiction::Entity::find().all(db).await?;
    let seeds = cfg
        .sanctioned_countries
        .iter()
        .map(|c| (c.clone(), "sanctioned", "Seed: OFAC-style sanctioned-jurisdiction list from the Synthetic KYC benchmark; replace with your own sanctions programme"))
        .chain(cfg.very_high_risk_countries.iter().map(|c| (c.clone(), "very_high", "Seed: jurisdictions with ~23% suspicious-flag rate vs ~1% elsewhere in the AML_Suspicious benchmark; replace with your own risk assessment / FATF lists")))
        .chain(cfg.high_risk_countries.iter().map(|c| (c.clone(), "high", "Seed list derived from the SAML-D benchmark skew; replace with your own risk assessment / FATF lists")));
    for (c, level, reason) in seeds {
        if !jur.iter().any(|j| j.country_code == c) {
            let m = jurisdiction::ActiveModel {
                country_code: Set(c.clone()),
                name: Set(country_name(&c).into()),
                risk_level: Set(level.into()),
                reason: Set(Some(reason.into())),
            }
            .insert(db)
            .await?;
            jur.push(m);
        }
    }
    cfg.sanctioned_countries = jur.iter().filter(|j| j.risk_level == "sanctioned").map(|j| j.country_code.clone()).collect();
    cfg.very_high_risk_countries = jur.iter().filter(|j| j.risk_level == "very_high").map(|j| j.country_code.clone()).collect();
    cfg.high_risk_countries = jur.iter().filter(|j| j.risk_level == "high").map(|j| j.country_code.clone()).collect();
    cfg.medium_risk_countries = jur.iter().filter(|j| j.risk_level == "medium").map(|j| j.country_code.clone()).collect();
    Ok(cfg)
}

/// Build an engine from the database: configuration, watchlist, and the recent transaction window.
pub async fn load_engine(db: &DatabaseConnection, hydrate_days: i64) -> Result<Engine, ApiError> {
    let cfg = load_config(db).await?;
    let mut engine = Engine::new(cfg);
    for w in watchlist::Entity::find().all(db).await? {
        engine.set_watch(w.account_number, true);
    }
    let pool = db.get_postgres_connection_pool();
    for r in sqlx::query(
        "SELECT a.account_number, c.risk_rating, c.pep, c.sector, c.annual_income_gbp::float8 AS income \
         FROM accounts a JOIN customers c ON c.id = a.customer_id",
    )
    .fetch_all(pool)
    .await?
    {
        if let Some(risk) = CustomerRisk::parse(&r.get::<String, _>("risk_rating")) {
            engine.set_profile(
                r.get("account_number"),
                &Profile {
                    risk,
                    pep: r.get("pep"),
                    annual_income_gbp: r.get::<Option<f64>, _>("income"),
                    trade_exposed: r.get::<Option<String>, _>("sector").as_deref().map(aml_engine::is_trade_sector).unwrap_or(false),
                },
            );
        }
    }
    let mut rows = sqlx::query(
        "SELECT id, EXTRACT(EPOCH FROM occurred_at)::bigint AS ts, sender_account, receiver_account, \
                amount_gbp::float8 AS gbp, payment_type, sender_country, receiver_country, \
                (payment_currency <> received_currency) AS xc \
         FROM transactions \
         WHERE occurred_at >= COALESCE((SELECT max(occurred_at) FROM transactions), now()) - make_interval(days => $1::int) \
         ORDER BY occurred_at, id",
    )
    .bind(hydrate_days as i32)
    .fetch(pool);
    let mut n = 0usize;
    while let Some(row) = rows.try_next().await? {
        let (Some(pt), Some(sc), Some(rc)) = (
            PaymentType::parse(&row.get::<String, _>("payment_type")),
            Country::parse(&row.get::<String, _>("sender_country")),
            Country::parse(&row.get::<String, _>("receiver_country")),
        ) else {
            continue;
        };
        let t = TxnIn {
            ts: row.get("ts"),
            sender: row.get("sender_account"),
            receiver: row.get("receiver_account"),
            amount_gbp: row.get("gbp"),
            payment_type: pt,
            sender_country: sc,
            receiver_country: rc,
            cross_currency: row.get("xc"),
        };
        engine.ingest(&t, row.get::<i64, _>("id") as u32);
        n += 1;
    }
    tracing::info!(transactions = n, accounts = engine.account_count(), "engine hydrated");
    Ok(engine)
}

