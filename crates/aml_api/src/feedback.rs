//! Connectome memory feedback loop, API side: capture analyst outcomes, learn from them, expose recommendations.
//!
//! `Analyst decision -> case outcome -> SAR filed? -> false positive? -> reinforcement signal` is recorded in
//! `feedback_events` whenever an alert reaches a terminal status (directly or through case closure). The learning
//! itself lives in the pure `aml_feedback` crate; this module only reads and writes the database and enforces
//! that nothing changes without an administrator.

use crate::{app::AppState, auth::Actor, error::{ApiError, ApiResult}, routes::audit};
use aml_agent::{rank, suggestions, Apply, PriorityConfig, QueueItem};
use aml_engine::config::ALL_RULES;
use aml_feedback::{analyse, advise, Calibrator, Decision, FeedbackEvent, MemoryConfig, OutcomeMemory, RuleHit, RuleLearnerConfig, ThresholdConfig};
use axum::{extract::{Path, Query, State}, Extension, Json};
use chrono::Utc;
use sea_orm::{sqlx::Row, ConnectionTrait, DbBackend, Statement, Value as SV};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const CAPTURE_SQL: &str = "INSERT INTO feedback_events (alert_id, account, decision, score, rules, graph_risk, decided_by, case_id, decided_at) \
 SELECT a.id, a.subject_account, \
        CASE a.status WHEN 'sar_filed' THEN 'sar_filed' WHEN 'closed_confirmed' THEN 'confirmed' ELSE 'false_positive' END, \
        a.score, \
        COALESCE((SELECT jsonb_agg(jsonb_build_object('rule', h->>'rule', 'weight', COALESCE((h->>'weight')::float8, 0))) \
                    FROM jsonb_array_elements(CASE WHEN jsonb_typeof(a.explanation->'hits') = 'array' THEN a.explanation->'hits' ELSE '[]'::jsonb END) h), '[]'::jsonb), \
        (SELECT g.risk FROM graph_risk g WHERE g.account = a.subject_account), \
        $1, a.case_id, now() \
   FROM alerts a WHERE a.status IN ('sar_filed','closed_confirmed','closed_false_positive') AND ";

/// Record the outcome of one alert (call after it reached a terminal status).
pub async fn capture_alert<C: ConnectionTrait>(db: &C, alert_id: i64, actor: &str) -> ApiResult<()> {
    let sql = format!("{CAPTURE_SQL} a.id = $2 ON CONFLICT (alert_id) DO UPDATE SET decision = EXCLUDED.decision, decided_by = EXCLUDED.decided_by, decided_at = now()");
    db.execute(Statement::from_sql_and_values(DbBackend::Postgres, sql, vec![SV::from(actor.to_string()), SV::from(alert_id)])).await?;
    Ok(())
}

/// Record the outcome of every terminal alert of a case.
pub async fn capture_case<C: ConnectionTrait>(db: &C, case_id: i64, actor: &str) -> ApiResult<()> {
    let sql = format!("{CAPTURE_SQL} a.case_id = $2 ON CONFLICT (alert_id) DO UPDATE SET decision = EXCLUDED.decision, decided_by = EXCLUDED.decided_by, decided_at = now()");
    db.execute(Statement::from_sql_and_values(DbBackend::Postgres, sql, vec![SV::from(actor.to_string()), SV::from(case_id)])).await?;
    Ok(())
}

/// An administrator reopened the alert: its earlier decision no longer stands.
pub async fn retract_alert<C: ConnectionTrait>(db: &C, alert_id: i64) -> ApiResult<()> {
    db.execute(Statement::from_sql_and_values(DbBackend::Postgres, "DELETE FROM feedback_events WHERE alert_id = $1", vec![SV::from(alert_id)])).await?;
    Ok(())
}

pub async fn load_events(st: &AppState, since_days: i64) -> ApiResult<Vec<FeedbackEvent>> {
    let rows = sea_orm::sqlx::query(
        "SELECT alert_id, account, decision, score::int AS score, rules, graph_risk, decided_by, case_id, EXTRACT(EPOCH FROM decided_at)::bigint AS ts \
         FROM feedback_events WHERE decided_at >= now() - make_interval(days => $1::int) ORDER BY decided_at",
    )
    .bind(since_days.clamp(1, 3650) as i32)
    .fetch_all(st.db.get_postgres_connection_pool())
    .await?;
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let Some(decision) = Decision::parse(&r.get::<String, _>("decision")) else { continue };
        let rules: Vec<RuleHit> = r
            .get::<Value, _>("rules")
            .as_array()
            .map(|a| a.iter().filter_map(|h| Some(RuleHit { rule: h.get("rule")?.as_str()?.to_string(), weight: h.get("weight").and_then(Value::as_f64).unwrap_or(0.0) })).collect())
            .unwrap_or_default();
        out.push(FeedbackEvent {
            alert_id: r.get("alert_id"),
            account: r.get("account"),
            decided_at: r.get("ts"),
            score: r.get::<i32, _>("score").clamp(0, 100) as u8,
            rules,
            graph_risk: r.get("graph_risk"),
            decision,
            decided_by: r.get("decided_by"),
            case_id: r.get("case_id"),
        });
    }
    Ok(out)
}

pub async fn current_weights(st: &AppState) -> BTreeMap<String, f64> {
    let eng = st.engine.lock().await;
    ALL_RULES.iter().map(|c| (c.to_string(), eng.config().rule(c).weight)).collect()
}

#[derive(Deserialize)]
pub struct WindowQ {
    since_days: Option<i64>,
}

/// What the loop has learned so far, without storing or changing anything.
pub async fn summary(State(st): State<AppState>, Query(q): Query<WindowQ>) -> ApiResult<Json<Value>> {
    let events = load_events(&st, q.since_days.unwrap_or(365)).await?;
    let weights = current_weights(&st).await;
    let threshold = st.engine.lock().await.alert_threshold();
    let rules = analyse(&events, &weights, &RuleLearnerConfig::default());
    let thr = advise(&events, threshold, &ThresholdConfig::default());
    let by = |d: &str| events.iter().filter(|e| e.decision.as_str() == d).count();
    let sugg = suggestions(&rules, &thr);
    Ok(Json(json!({
        "events": events.len(),
        "sar_filed": by("sar_filed"), "confirmed": by("confirmed"), "false_positive": by("false_positive"),
        "rules": rules, "threshold": thr, "suggestions": sugg,
    })))
}

/// Turn the current findings into stored, pending recommendations (older pending ones are superseded).
pub async fn recompute(State(st): State<AppState>, Extension(a): Extension<Actor>, Query(q): Query<WindowQ>) -> ApiResult<Json<Value>> {
    a.require_admin()?;
    let events = load_events(&st, q.since_days.unwrap_or(365)).await?;
    let weights = current_weights(&st).await;
    let threshold = st.engine.lock().await.alert_threshold();
    let rules = analyse(&events, &weights, &RuleLearnerConfig::default());
    let thr = advise(&events, threshold, &ThresholdConfig::default());
    let sugg = suggestions(&rules, &thr);
    let pool = st.db.get_postgres_connection_pool();
    let mut tx = pool.begin().await?;
    sea_orm::sqlx::query("UPDATE rule_recommendations SET status='superseded' WHERE status='pending'").execute(&mut *tx).await?;
    let mut ids = vec![];
    for s in &sugg {
        let id: i64 = sea_orm::sqlx::query("INSERT INTO rule_recommendations (kind, title, detail, apply) VALUES ($1,$2,$3,$4) RETURNING id")
            .bind(&s.kind)
            .bind(&s.title)
            .bind(&s.detail)
            .bind(serde_json::to_value(&s.apply).unwrap_or(Value::Null))
            .fetch_one(&mut *tx)
            .await?
            .get(0);
        ids.push(id);
    }
    tx.commit().await?;
    audit(&st, &a, "feedback.recompute", "feedback", "all", json!({"events": events.len(), "recommendations": ids})).await;
    Ok(Json(json!({"events": events.len(), "recommendation_ids": ids, "suggestions": sugg, "caveats": rules.caveats})))
}

#[derive(Deserialize)]
pub struct RecQ {
    status: Option<String>,
}

pub async fn list_recommendations(State(st): State<AppState>, Query(q): Query<RecQ>) -> ApiResult<Json<Value>> {
    let rows = sea_orm::sqlx::query(
        "SELECT id, kind, title, detail, apply, status, created_at, decided_by, decided_at FROM rule_recommendations \
         WHERE ($1::text IS NULL OR status = $1) ORDER BY id DESC LIMIT 200",
    )
    .bind(q.status)
    .fetch_all(st.db.get_postgres_connection_pool())
    .await?;
    let items: Vec<Value> = rows
        .iter()
        .map(|r| json!({
            "id": r.get::<i64, _>("id"), "kind": r.get::<String, _>("kind"), "title": r.get::<String, _>("title"),
            "detail": r.get::<String, _>("detail"), "apply": r.get::<Option<Value>, _>("apply"), "status": r.get::<String, _>("status"),
            "created_at": r.get::<chrono::DateTime<Utc>, _>("created_at"), "decided_by": r.get::<Option<String>, _>("decided_by"),
            "decided_at": r.get::<Option<chrono::DateTime<Utc>>, _>("decided_at"),
        }))
        .collect();
    Ok(Json(json!(items)))
}

async fn pending(st: &AppState, id: i64) -> ApiResult<(String, Option<Value>)> {
    let r = sea_orm::sqlx::query("SELECT kind, apply, status FROM rule_recommendations WHERE id=$1")
        .bind(id)
        .fetch_optional(st.db.get_postgres_connection_pool())
        .await?
        .ok_or_else(|| ApiError::NotFound(format!("recommendation {id}")))?;
    let status: String = r.get("status");
    if status != "pending" {
        return Err(ApiError::Conflict(format!("recommendation {id} is already {status}")));
    }
    Ok((r.get("kind"), r.get("apply")))
}

pub async fn reject(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(id): Path<i64>) -> ApiResult<Json<Value>> {
    a.require_admin()?;
    pending(&st, id).await?;
    sea_orm::sqlx::query("UPDATE rule_recommendations SET status='rejected', decided_by=$2, decided_at=now() WHERE id=$1")
        .bind(id)
        .bind(&a.name)
        .execute(st.db.get_postgres_connection_pool())
        .await?;
    audit(&st, &a, "feedback.reject", "recommendation", id, json!({})).await;
    Ok(Json(json!({"id": id, "status": "rejected"})))
}

/// Apply a pending recommendation: a bounded rule-weight change or an alert-threshold change. Admin only, audited.
pub async fn apply(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(id): Path<i64>) -> ApiResult<Json<Value>> {
    a.require_admin()?;
    let (kind, apply) = pending(&st, id).await?;
    let ap: Apply = serde_json::from_value(apply.unwrap_or(Value::Null)).map_err(|_| ApiError::Conflict(format!("recommendation {id} ({kind}) is advisory and has nothing to apply")))?;
    let pool = st.db.get_postgres_connection_pool();
    let detail = match &ap {
        Apply::RuleWeight { rule, weight } => {
            if !ALL_RULES.contains(&rule.as_str()) || !(0.0..0.99).contains(weight) {
                return Err(ApiError::Unprocessable(vec!["recommendation holds an invalid rule or weight".into()]));
            }
            let before: Option<f64> = sea_orm::sqlx::query("SELECT weight FROM rule_configs WHERE code=$1").bind(rule).fetch_optional(pool).await?.map(|r| r.get(0));
            sea_orm::sqlx::query("UPDATE rule_configs SET weight=$2, updated_at=now(), updated_by=$3 WHERE code=$1").bind(rule).bind(*weight).bind(&a.name).execute(pool).await?;
            json!({"rule": rule, "before": before, "after": weight})
        }
        Apply::AlertThreshold { value } => {
            if !(1..=100).contains(value) {
                return Err(ApiError::Unprocessable(vec!["threshold must be 1-100".into()]));
            }
            sea_orm::sqlx::query("UPDATE rule_configs SET params = jsonb_build_object('alert_threshold', $2::int), updated_at=now(), updated_by=$3 WHERE code='ENGINE'")
                .bind(*value as i32)
                .bind(&a.name)
                .execute(pool)
                .await?;
            json!({"alert_threshold": value})
        }
    };
    let cfg = crate::ingest::load_config(&st.db).await?;
    st.engine.lock().await.set_config(cfg);
    sea_orm::sqlx::query("UPDATE rule_recommendations SET status='applied', decided_by=$2, decided_at=now() WHERE id=$1").bind(id).bind(&a.name).execute(pool).await?;
    audit(&st, &a, "feedback.apply", "recommendation", id, detail.clone()).await;
    Ok(Json(json!({"id": id, "status": "applied", "change": detail})))
}

/// Review queue ordered by calibrated probability, graph risk, exposure, age and outcome memory.
pub async fn queue(State(st): State<AppState>, Query(q): Query<WindowQ>) -> ApiResult<Json<Value>> {
    let events = load_events(&st, q.since_days.unwrap_or(365)).await?;
    let now = Utc::now().timestamp();
    let cal = Calibrator::fit(&events.iter().map(|e| (e.score as f64, e.label(), e.weight())).collect::<Vec<_>>()).filter(|_| events.len() >= 50);
    let mcfg = MemoryConfig::default();
    let mem = OutcomeMemory::from_events(&events, now, &mcfg);
    let rows = sea_orm::sqlx::query(
        "SELECT a.id, a.subject_account, a.score::int AS score, a.severity, a.rule_codes, \
                COALESCE((a.explanation->>'amount_gbp')::float8, 0) AS amount, \
                (EXTRACT(EPOCH FROM (now() - a.created_at))/86400.0)::float8 AS age, g.risk \
           FROM alerts a LEFT JOIN graph_risk g ON g.account = a.subject_account \
          WHERE a.status IN ('open','under_review','escalated') ORDER BY a.score DESC, a.id DESC LIMIT 500",
    )
    .fetch_all(st.db.get_postgres_connection_pool())
    .await?;
    let items: Vec<QueueItem> = rows
        .iter()
        .map(|r| {
            let rules: Vec<String> = r.get::<Value, _>("rule_codes").as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
            let account: i64 = r.get("subject_account");
            let refs: Vec<&str> = rules.iter().map(String::as_str).collect();
            let adj = mem.adjust(account, &refs, &mcfg);
            let score = r.get::<i32, _>("score").clamp(0, 100) as u8;
            QueueItem {
                alert_id: r.get("id"),
                account,
                score,
                severity: r.get("severity"),
                rules,
                amount_gbp: r.get::<f64, _>("amount"),
                age_days: r.get::<f64, _>("age"),
                graph_risk: r.get("risk"),
                calibrated: cal.as_ref().map(|c| c.predict(score as f64)),
                memory_factor: adj.factor,
                memory_reason: adj.reason,
            }
        })
        .collect();
    let ranked = rank(&items, &PriorityConfig::default());
    Ok(Json(json!({
        "calibrated": cal.is_some(),
        "note": if cal.is_some() { "ordered by calibrated probability" } else { "fewer than 50 reviewed alerts: ordered by engine score until analyst outcomes accumulate" },
        "queue": ranked,
    })))
}
