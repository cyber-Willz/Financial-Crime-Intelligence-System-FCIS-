//! Bidirectional LLM analyst agent, API side: builds the evidence packet and the graph neighbourhood from the
//! database, runs `aml_agent`, and records every output in `agent_runs`.
//!
//! Down: alert/case data, rule activations, graph risk, sanctions flags, ownership (optional file), analyst
//! notes, earlier outcomes, calibrated probability. Up: the agent's whitelisted graph queries are answered from a
//! pre-fetched neighbourhood (subject + its ten biggest counterparties), so a model can never reach the database.

use crate::{app::AppState, auth::Actor, error::{ApiError, ApiResult}};
use aml_agent::evidence::*;
use aml_agent::{risk_summary, Agent, CaseFacts, Flow, MemGraph};
use aml_entity::{account, alert, case, case_note, customer};
use aml_feedback::{Calibrator, MemoryConfig, OutcomeMemory};
use axum::{extract::{Path, Query, State}, Extension, Json};
use chrono::{DateTime, Utc};
use sea_orm::{sqlx::Row, ColumnTrait, EntityTrait, QueryFilter, QueryOrder, QuerySelect};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

fn d(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%d").to_string()
}

fn dec(x: sea_orm::prelude::Decimal) -> f64 {
    x.to_string().parse().unwrap_or(0.0)
}

pub async fn build_evidence(st: &AppState, subject: i64, al: Option<&alert::Model>) -> ApiResult<Evidence> {
    let pool = st.db.get_postgres_connection_pool();
    let mut ev = Evidence { subject, ..Default::default() };
    if let Some(a) = al {
        let hits = a.explanation.get("hits").and_then(Value::as_array).cloned().unwrap_or_default();
        ev.alert = Some(AlertFacts {
            id: a.id,
            score: a.score.clamp(0, 100) as u8,
            severity: a.severity.clone(),
            status: a.status.clone(),
            txn_count: a.txn_count,
            created_at: d(a.created_at),
            rules: hits.iter().filter_map(|h| Some(RuleFact { code: h.get("rule")?.as_str()?.to_string(), weight: h.get("weight").and_then(Value::as_f64).unwrap_or(0.0), detail: h.get("detail").and_then(Value::as_str).unwrap_or("").to_string() })).collect(),
        });
    }
    if let Some(ac) = account::Entity::find_by_id(subject).one(&st.db).await? {
        ev.account = Some(AccountFacts {
            account: subject,
            country: ac.country.clone(),
            risk_level: ac.risk_level.clone(),
            risk_score: ac.risk_score,
            alert_count: ac.alert_count,
            watchlisted: ac.watchlisted,
            first_seen: d(ac.first_seen),
            last_seen: d(ac.last_seen),
            txn_sent: ac.txn_sent_count,
            txn_received: ac.txn_received_count,
            total_sent_gbp: dec(ac.total_sent),
            total_received_gbp: dec(ac.total_received),
        });
        ev.sanctions.watchlist_hit = ac.watchlisted;
        if ac.watchlisted {
            ev.sanctions.watchlist_name = Some("internal watchlist".into());
        }
        if let Some(cid) = ac.customer_id {
            if let Some(c) = customer::Entity::find_by_id(cid).one(&st.db).await? {
                ev.customer = Some(CustomerFacts { reference: c.external_ref, customer_type: c.customer_type, country: c.country, sector: c.sector, risk_rating: c.risk_rating, pep: c.pep, annual_income_gbp: c.annual_income_gbp.map(dec) });
            }
        }
    }
    if let Some(a) = al {
        let rows = sea_orm::sqlx::query(
            "SELECT id, occurred_at, sender_account, receiver_account, amount_gbp::float8 AS amt, payment_type, sender_country, receiver_country \
               FROM transactions WHERE id BETWEEN $1 AND $2 AND (sender_account = $3 OR receiver_account = $3) AND flagged ORDER BY id LIMIT 15",
        )
        .bind(a.first_txn_id)
        .bind(a.last_txn_id)
        .bind(subject)
        .fetch_all(pool)
        .await?;
        ev.transactions = rows
            .iter()
            .map(|r| TxnFact { id: r.get("id"), at: d(r.get("occurred_at")), sender: r.get("sender_account"), receiver: r.get("receiver_account"), amount_gbp: r.get("amt"), payment_type: r.get("payment_type"), sender_country: r.get("sender_country"), receiver_country: r.get("receiver_country") })
            .collect();
        if let Some(cid) = a.case_id {
            for n in case_note::Entity::find().filter(case_note::Column::CaseId.eq(cid)).order_by_asc(case_note::Column::Id).limit(10).all(&st.db).await? {
                ev.notes.push(NoteFact { author: n.author, at: d(n.created_at), body: n.body });
            }
        }
    }
    if let Some(r) = sea_orm::sqlx::query("SELECT risk, window_start FROM graph_risk WHERE account = $1").bind(subject).fetch_optional(pool).await? {
        ev.graph = GraphFacts { risk: Some(r.get("risk")), window: r.get::<Option<DateTime<Utc>>, _>("window_start").map(d) };
    }
    let cs = sea_orm::sqlx::query(
        "SELECT DISTINCT x.c FROM (SELECT receiver_country AS c FROM (SELECT receiver_country FROM transactions WHERE sender_account = $1 ORDER BY occurred_at DESC LIMIT 2000) a \
         UNION SELECT sender_country FROM (SELECT sender_country FROM transactions WHERE receiver_account = $1 ORDER BY occurred_at DESC LIMIT 2000) b) x \
         JOIN jurisdictions j ON j.country_code = x.c WHERE j.risk_level = 'sanctioned' ORDER BY 1",
    )
    .bind(subject)
    .fetch_all(pool)
    .await?;
    ev.sanctions.sanctioned_jurisdiction_countries = cs.iter().map(|r| r.get::<String, _>(0)).collect();
    // earlier outcomes (excluding the alert under review)
    let events = crate::feedback::load_events(st, 3650).await?;
    let own: Vec<_> = events.iter().filter(|e| e.account == subject && Some(e.alert_id) != al.map(|a| a.id)).collect();
    ev.history = HistoryFacts {
        prior_alerts: own.len() as u32,
        prior_sar: own.iter().filter(|e| e.decision.as_str() == "sar_filed").count() as u32,
        prior_confirmed: own.iter().filter(|e| e.decision.as_str() == "confirmed").count() as u32,
        prior_false_positive: own.iter().filter(|e| e.decision.as_str() == "false_positive").count() as u32,
    };
    let mcfg = MemoryConfig::default();
    let mem = OutcomeMemory::from_events(&events, Utc::now().timestamp(), &mcfg);
    let rules = ev.rule_codes().iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let refs: Vec<&str> = rules.iter().map(String::as_str).collect();
    let adj = mem.adjust(subject, &refs, &mcfg);
    let cal = if events.len() >= 50 { Calibrator::fit(&events.iter().map(|e| (e.score as f64, e.label(), e.weight())).collect::<Vec<_>>()) } else { None };
    if let (Some(c), Some(a)) = (&cal, al) {
        ev.feedback = Some(FeedbackFacts { calibrated_probability: Some(c.predict(a.score as f64)), priority_factor: adj.factor, factor_reason: adj.reason });
    } else if (adj.factor - 1.0).abs() > 1e-9 {
        ev.feedback = Some(FeedbackFacts { calibrated_probability: None, priority_factor: adj.factor, factor_reason: adj.reason });
    }
    ev.ownership = st.ownership.get(&subject).cloned();
    Ok(ev)
}

/// Pre-fetch the subject's flows and those of its ten largest counterparties.
pub async fn build_graph(st: &AppState, subject: i64) -> ApiResult<MemGraph> {
    let pool = st.db.get_postgres_connection_pool();
    const Q: &str = "SELECT id, EXTRACT(EPOCH FROM occurred_at)::bigint AS ts, sender_account, receiver_account, amount_gbp::float8 AS amt, payment_type, sender_country, receiver_country \
                     FROM transactions WHERE sender_account = $1 OR receiver_account = $1 ORDER BY occurred_at DESC LIMIT $2";
    let to_flows = |rows: Vec<sea_orm::sqlx::postgres::PgRow>| -> Vec<Flow> {
        rows.iter().map(|r| Flow { id: r.get("id"), ts: r.get("ts"), from: r.get("sender_account"), to: r.get("receiver_account"), amount_gbp: r.get("amt"), ptype: r.get("payment_type"), from_cc: r.get("sender_country"), to_cc: r.get("receiver_country") }).collect()
    };
    let first = to_flows(sea_orm::sqlx::query(Q).bind(subject).bind(1000i64).fetch_all(pool).await?);
    let mut vol: BTreeMap<i64, f64> = BTreeMap::new();
    for f in &first {
        *vol.entry(if f.from == subject { f.to } else { f.from }).or_default() += f.amount_gbp;
    }
    let mut top: Vec<(i64, f64)> = vol.into_iter().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut all = first;
    let peers: Vec<i64> = top.iter().take(10).map(|x| x.0).collect();
    for p in &peers {
        all.extend(to_flows(sea_orm::sqlx::query(Q).bind(*p).bind(300i64).fetch_all(pool).await?));
    }
    let mut g = MemGraph::new(all);
    for (k, v) in st.ownership.iter() {
        if *k == subject || peers.contains(k) {
            g.ownership.insert(*k, v.clone());
        }
    }
    let mut ids = peers.clone();
    ids.push(subject);
    let rows = sea_orm::sqlx::query("SELECT account, decision, count(*)::int AS n FROM feedback_events WHERE account = ANY($1) GROUP BY 1, 2").bind(&ids).fetch_all(pool).await?;
    let mut h: HashMap<i64, HistoryFacts> = HashMap::new();
    for r in rows {
        let e = h.entry(r.get("account")).or_default();
        let n = r.get::<i32, _>("n") as u32;
        e.prior_alerts += n;
        match r.get::<String, _>("decision").as_str() {
            "sar_filed" => e.prior_sar += n,
            "confirmed" => e.prior_confirmed += n,
            _ => e.prior_false_positive += n,
        }
    }
    g.history = h;
    Ok(g)
}

#[derive(Deserialize)]
pub struct Mode {
    /// skip the language model even when one is attached
    deterministic: Option<bool>,
}

async fn run<T: Send + 'static>(st: &AppState, deterministic: bool, f: impl FnOnce(&mut Agent) -> T + Send + 'static) -> ApiResult<T> {
    let agent = st.agent.clone();
    let use_llm = !deterministic && agent.lock().map(|a| a.has_llm()).unwrap_or(false);
    if !use_llm {
        return Ok(f(&mut Agent::deterministic()));
    }
    tokio::task::spawn_blocking(move || {
        let mut g = agent.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut g)
    })
    .await
    .map_err(|e| ApiError::Internal(anyhow::anyhow!("agent task failed: {e}")))
}

async fn record<T: Serialize>(st: &AppState, a: &Actor, kind: &str, subject: String, digest: String, out: &T, prov: &aml_agent::Provenance) {
    let r = sea_orm::sqlx::query(
        "INSERT INTO agent_runs (kind, subject, actor, evidence_digest, llm_model, llm_used, grounding_ok, fallback_reason, output) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
    )
    .bind(kind)
    .bind(subject)
    .bind(&a.name)
    .bind(digest)
    .bind(&prov.model)
    .bind(prov.llm_used)
    .bind(prov.grounding.ok)
    .bind(&prov.fallback_reason)
    .bind(serde_json::to_value(out).unwrap_or(Value::Null))
    .execute(st.db.get_postgres_connection_pool())
    .await;
    if let Err(e) = r {
        tracing::error!(error = ?e, "agent_runs write failed");
    }
}

pub async fn status(State(st): State<AppState>) -> Json<Value> {
    let (has, n) = st.agent.lock().map(|a| (a.has_llm(), 0)).unwrap_or((false, 0));
    Json(json!({"llm_attached": has, "mode": if has { "language model proposes; grounding gate decides" } else { "deterministic analyst only" }, "ownership_records": st.ownership.len() + n}))
}

pub async fn explain(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(id): Path<i64>, Query(m): Query<Mode>) -> ApiResult<Json<Value>> {
    a.require_triage()?;
    let al = alert::Entity::find_by_id(id).one(&st.db).await?.ok_or_else(|| ApiError::NotFound(format!("alert {id}")))?;
    let ev = build_evidence(&st, al.subject_account, Some(&al)).await?;
    let g = build_graph(&st, al.subject_account).await?;
    let digest = ev.digest();
    let ev2 = ev.clone();
    let ex = run(&st, m.deterministic.unwrap_or(false), move |ag| ag.explain_alert(&ev2, &g)).await?;
    record(&st, &a, "explain", id.to_string(), digest, &ex, &ex.provenance).await;
    Ok(Json(json!(ex)))
}

#[derive(Deserialize)]
pub struct Ask {
    alert_id: Option<i64>,
    account: Option<i64>,
    question: String,
}

pub async fn investigate(State(st): State<AppState>, Extension(a): Extension<Actor>, Query(m): Query<Mode>, Json(q): Json<Ask>) -> ApiResult<Json<Value>> {
    a.require_triage()?;
    if q.question.trim().is_empty() || q.question.len() > 500 {
        return Err(ApiError::Unprocessable(vec!["question must be 1-500 characters".into()]));
    }
    let al = match q.alert_id {
        Some(id) => Some(alert::Entity::find_by_id(id).one(&st.db).await?.ok_or_else(|| ApiError::NotFound(format!("alert {id}")))?),
        None => None,
    };
    let subject = al.as_ref().map(|x| x.subject_account).or(q.account).ok_or_else(|| ApiError::Unprocessable(vec!["give alert_id or account".into()]))?;
    let ev = build_evidence(&st, subject, al.as_ref()).await?;
    let g = build_graph(&st, subject).await?;
    let digest = ev.digest();
    let question = q.question.clone();
    let inv = run(&st, m.deterministic.unwrap_or(false), move |ag| ag.investigate(&question, &ev, &g)).await?;
    record(&st, &a, "investigate", subject.to_string(), digest, &inv, &inv.provenance).await;
    Ok(Json(json!(inv)))
}

pub async fn sar_draft(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(id): Path<i64>, Query(m): Query<Mode>) -> ApiResult<Json<Value>> {
    a.require_triage()?;
    let c = case::Entity::find_by_id(id).one(&st.db).await?.ok_or_else(|| ApiError::NotFound(format!("case {id}")))?;
    let alerts = alert::Entity::find().filter(alert::Column::CaseId.eq(id)).order_by_desc(alert::Column::Score).limit(10).all(&st.db).await?;
    if alerts.is_empty() {
        return Err(ApiError::Conflict("the case has no alerts to base a draft on".into()));
    }
    let mut evs = vec![];
    for al in &alerts {
        evs.push(build_evidence(&st, al.subject_account, Some(al)).await?);
    }
    let g = build_graph(&st, alerts[0].subject_account).await?;
    let notes: Vec<NoteFact> = case_note::Entity::find().filter(case_note::Column::CaseId.eq(id)).order_by_asc(case_note::Column::Id).limit(30).all(&st.db).await?.into_iter().map(|n| NoteFact { author: n.author, at: d(n.created_at), body: n.body }).collect();
    let facts = CaseFacts { case_id: id, title: c.title.clone(), priority: c.priority.clone(), status: c.status.clone(), notes };
    let digest = evs.iter().map(|e| e.digest()).collect::<Vec<_>>().join(",");
    let draft = run(&st, m.deterministic.unwrap_or(false), move |ag| ag.sar_draft(&facts, &evs, &g)).await?;
    record(&st, &a, "sar_draft", id.to_string(), digest, &draft, &draft.provenance).await;
    Ok(Json(json!(draft)))
}

/// Counts over the open alerts (computed in code; no model involved).
pub async fn summary(State(st): State<AppState>) -> ApiResult<Json<Value>> {
    let als = alert::Entity::find().filter(alert::Column::Status.is_in(["open", "under_review", "escalated"])).order_by_desc(alert::Column::Score).limit(100).all(&st.db).await?;
    let mut evs = vec![];
    for al in &als {
        evs.push(build_evidence(&st, al.subject_account, Some(al)).await?);
    }
    Ok(Json(json!(risk_summary(&evs))))
}
