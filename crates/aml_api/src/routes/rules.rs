use super::audit;
use crate::{app::AppState, auth::Actor, error::{ApiError, ApiResult}, ingest};
use aml_engine::config::ALL_RULES;
use aml_entity::{jurisdiction, rule_config};
use axum::{extract::{Path, State}, Extension, Json};
use chrono::Utc;
use sea_orm::{ActiveModelTrait, EntityTrait, Set};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub async fn list(State(st): State<AppState>) -> Json<Value> {
    let eng = st.engine.lock().await;
    Json(serde_json::to_value(eng.config()).unwrap_or(Value::Null))
}

#[derive(Deserialize)]
pub struct RuleUpdate {
    enabled: Option<bool>,
    weight: Option<f64>,
    params: Option<BTreeMap<String, f64>>,
}

/// Rule changes take effect for transactions scored from now on; history is not re-scored.
pub async fn update(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(code): Path<String>, Json(u): Json<RuleUpdate>) -> ApiResult<Json<Value>> {
    a.require_admin()?;
    if !ALL_RULES.contains(&code.as_str()) {
        return Err(ApiError::NotFound(format!("unknown rule {code}")));
    }
    let row = rule_config::Entity::find_by_id(code.clone()).one(&st.db).await?.ok_or_else(|| ApiError::NotFound(code.clone()))?;
    let mut errs = vec![];
    if let Some(w) = u.weight {
        if !(0.0..0.99).contains(&w) {
            errs.push("weight must be in [0, 0.99)".to_string());
        }
    }
    // start from the *effective* parameters so newly introduced parameters can be set on older databases
    let mut params: BTreeMap<String, f64> = st.engine.lock().await.config().rule(&code).params;
    if let Some(p) = &u.params {
        for (k, v) in p {
            if !params.contains_key(k) {
                errs.push(format!("unknown parameter '{k}' for {code}"));
            } else if !v.is_finite() || *v < 0.0 {
                errs.push(format!("parameter '{k}' must be a non-negative number"));
            } else {
                params.insert(k.clone(), *v);
            }
        }
    }
    if !errs.is_empty() {
        return Err(ApiError::Unprocessable(errs));
    }
    let before = json!({"enabled": row.enabled, "weight": row.weight, "params": row.params});
    let mut am: rule_config::ActiveModel = row.into();
    if let Some(e) = u.enabled {
        am.enabled = Set(e);
    }
    if let Some(w) = u.weight {
        am.weight = Set(w);
    }
    am.params = Set(serde_json::to_value(&params).unwrap());
    am.updated_at = Set(Utc::now());
    am.updated_by = Set(Some(a.name.clone()));
    let m = am.update(&st.db).await?;
    reload(&st).await?;
    audit(&st, &a, "rule.update", "rule", &code, json!({"before": before, "after": {"enabled": m.enabled, "weight": m.weight, "params": m.params}})).await;
    Ok(Json(json!(m)))
}

#[derive(Deserialize)]
pub struct Settings {
    alert_threshold: u8,
}

pub async fn settings(State(st): State<AppState>, Extension(a): Extension<Actor>, Json(s): Json<Settings>) -> ApiResult<Json<Value>> {
    a.require_admin()?;
    if !(1..=100).contains(&s.alert_threshold) {
        return Err(ApiError::Unprocessable(vec!["alert_threshold must be 1-100".into()]));
    }
    let row = rule_config::Entity::find_by_id("ENGINE").one(&st.db).await?.ok_or_else(|| ApiError::NotFound("ENGINE".into()))?;
    let mut am: rule_config::ActiveModel = row.into();
    am.params = Set(json!({"alert_threshold": s.alert_threshold}));
    am.updated_at = Set(Utc::now());
    am.updated_by = Set(Some(a.name.clone()));
    am.update(&st.db).await?;
    reload(&st).await?;
    audit(&st, &a, "settings.update", "engine", "alert_threshold", json!({"alert_threshold": s.alert_threshold})).await;
    Ok(Json(json!({"alert_threshold": s.alert_threshold})))
}

async fn reload(st: &AppState) -> ApiResult<()> {
    let cfg = ingest::load_config(&st.db).await?;
    st.engine.lock().await.set_config(cfg);
    Ok(())
}

pub async fn jurisdictions(State(st): State<AppState>) -> ApiResult<Json<Vec<jurisdiction::Model>>> {
    Ok(Json(jurisdiction::Entity::find().all(&st.db).await?))
}

#[derive(Deserialize)]
pub struct JurReq {
    name: Option<String>,
    risk_level: String,
    reason: Option<String>,
}

pub async fn set_jurisdiction(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(cc): Path<String>, Json(j): Json<JurReq>) -> ApiResult<Json<jurisdiction::Model>> {
    a.require_admin()?;
    let cc = cc.to_ascii_uppercase();
    if aml_engine::Country::parse(&cc).is_none() {
        return Err(ApiError::Unprocessable(vec!["country code must be ISO-3166 alpha-2".into()]));
    }
    if !["low", "medium", "high", "very_high", "sanctioned"].contains(&j.risk_level.as_str()) {
        return Err(ApiError::Unprocessable(vec!["risk_level must be low|medium|high|very_high|sanctioned".into()]));
    }
    let existing = jurisdiction::Entity::find_by_id(cc.clone()).one(&st.db).await?;
    let m = match existing {
        Some(e) => {
            let mut am: jurisdiction::ActiveModel = e.into();
            am.risk_level = Set(j.risk_level.clone());
            if let Some(n) = j.name { am.name = Set(n); }
            if j.reason.is_some() { am.reason = Set(j.reason); }
            am.update(&st.db).await?
        }
        None => jurisdiction::ActiveModel {
            country_code: Set(cc.clone()),
            name: Set(j.name.unwrap_or_else(|| cc.clone())),
            risk_level: Set(j.risk_level.clone()),
            reason: Set(j.reason),
        }
        .insert(&st.db)
        .await?,
    };
    reload(&st).await?;
    audit(&st, &a, "jurisdiction.set", "jurisdiction", &cc, json!({"risk_level": j.risk_level})).await;
    Ok(Json(m))
}

#[derive(Deserialize)]
pub struct ImportRule {
    enabled: Option<bool>,
    weight: Option<f64>,
    params: Option<BTreeMap<String, f64>>,
}

#[derive(Deserialize)]
pub struct ImportCfg {
    alert_threshold: Option<u8>,
    rules: BTreeMap<String, ImportRule>,
}

/// Apply a calibration produced by `aml-eval --fit-out` (all-or-nothing validation, then applied and audited).
pub async fn import(State(st): State<AppState>, Extension(a): Extension<Actor>, Json(c): Json<ImportCfg>) -> ApiResult<Json<Value>> {
    a.require_admin()?;
    let current = st.engine.lock().await.config().clone();
    let mut errs = vec![];
    let mut plan = vec![];
    for (code, r) in &c.rules {
        if !ALL_RULES.contains(&code.as_str()) {
            errs.push(format!("unknown rule {code}"));
            continue;
        }
        let cur = current.rule(code);
        if r.weight.is_some_and(|w| !(0.0..0.99).contains(&w)) {
            errs.push(format!("{code}: weight must be in [0, 0.99)"));
        }
        let mut params = cur.params.clone();
        for (k, v) in r.params.iter().flatten() {
            if !params.contains_key(k) || !v.is_finite() || *v < 0.0 {
                errs.push(format!("{code}: invalid parameter '{k}'"));
            } else {
                params.insert(k.clone(), *v);
            }
        }
        plan.push((code.clone(), r.enabled, r.weight, params));
    }
    if c.alert_threshold.is_some_and(|t| !(1..=100).contains(&t)) {
        errs.push("alert_threshold must be 1-100".into());
    }
    if !errs.is_empty() {
        return Err(ApiError::Unprocessable(errs));
    }
    for (code, enabled, weight, params) in &plan {
        let Some(row) = rule_config::Entity::find_by_id(code.clone()).one(&st.db).await? else { continue };
        let mut am: rule_config::ActiveModel = row.into();
        if let Some(e) = enabled { am.enabled = Set(*e); }
        if let Some(w) = weight { am.weight = Set(*w); }
        am.params = Set(serde_json::to_value(params).unwrap());
        am.updated_at = Set(Utc::now());
        am.updated_by = Set(Some(a.name.clone()));
        am.update(&st.db).await?;
    }
    if let Some(t) = c.alert_threshold {
        if let Some(row) = rule_config::Entity::find_by_id("ENGINE").one(&st.db).await? {
            let mut am: rule_config::ActiveModel = row.into();
            am.params = Set(json!({"alert_threshold": t}));
            am.updated_at = Set(Utc::now());
            am.update(&st.db).await?;
        }
    }
    reload(&st).await?;
    audit(&st, &a, "rules.import", "rule", "*", json!({"rules": plan.iter().map(|p| p.0.clone()).collect::<Vec<_>>(), "alert_threshold": c.alert_threshold})).await;
    Ok(Json(json!({"applied": plan.len(), "alert_threshold": c.alert_threshold})))
}
