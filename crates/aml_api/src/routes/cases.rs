use super::{audit, page};
use crate::{app::AppState, auth::Actor, error::{ApiError, ApiResult}};
use aml_entity::{alert, case, case_note};
use axum::{extract::{Path, Query, State}, http::StatusCode, Extension, Json};
use chrono::Utc;
use sea_orm::{
    sea_query::Expr, ActiveModelTrait, ColumnTrait, EntityTrait, QueryFilter, QueryOrder, QuerySelect, Set, TransactionTrait,
};
use serde::Deserialize;
use serde_json::{json, Value};

const PRIORITIES: [&str; 4] = ["low", "medium", "high", "critical"];

#[derive(Deserialize)]
pub struct New {
    title: String,
    description: Option<String>,
    priority: Option<String>,
    assigned_to: Option<String>,
    #[serde(default)]
    alert_ids: Vec<i64>,
}

async fn load(st: &AppState, id: i64) -> ApiResult<case::Model> {
    case::Entity::find_by_id(id).one(&st.db).await?.ok_or_else(|| ApiError::NotFound(format!("case {id}")))
}

async fn link<C: sea_orm::ConnectionTrait>(db: &C, case_id: i64, ids: &[i64]) -> ApiResult<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let found = alert::Entity::find().filter(alert::Column::Id.is_in(ids.to_vec())).all(db).await?;
    if found.len() != ids.len() {
        return Err(ApiError::Unprocessable(vec!["one or more alert_ids do not exist".into()]));
    }
    if let Some(a) = found.iter().find(|a| a.case_id.is_some_and(|c| c != case_id)) {
        return Err(ApiError::Conflict(format!("alert {} already belongs to case {}", a.id, a.case_id.unwrap())));
    }
    alert::Entity::update_many()
        .col_expr(alert::Column::CaseId, Expr::value(case_id))
        .col_expr(alert::Column::UpdatedAt, Expr::value(Utc::now()))
        .filter(alert::Column::Id.is_in(ids.to_vec()))
        .exec(db)
        .await?;
    alert::Entity::update_many()
        .col_expr(alert::Column::Status, Expr::value("under_review"))
        .filter(alert::Column::Id.is_in(ids.to_vec()))
        .filter(alert::Column::Status.eq("open"))
        .exec(db)
        .await?;
    Ok(())
}

pub async fn create(State(st): State<AppState>, Extension(a): Extension<Actor>, Json(n): Json<New>) -> ApiResult<(StatusCode, Json<Value>)> {
    a.require_triage()?;
    let title = n.title.trim().to_string();
    if title.is_empty() || title.len() > 200 {
        return Err(ApiError::Unprocessable(vec!["title must be 1-200 characters".into()]));
    }
    let prio = n.priority.unwrap_or_else(|| "medium".into());
    if !PRIORITIES.contains(&prio.as_str()) {
        return Err(ApiError::Unprocessable(vec![format!("priority must be one of {PRIORITIES:?}")]));
    }
    if n.alert_ids.len() > 100 {
        return Err(ApiError::Unprocessable(vec!["at most 100 alerts per request".into()]));
    }
    let tx = st.db.begin().await?;
    let c = case::ActiveModel {
        title: Set(title),
        description: Set(n.description),
        status: Set("open".into()),
        priority: Set(prio),
        assigned_to: Set(n.assigned_to),
        created_by: Set(a.name.clone()),
        created_at: Set(Utc::now()),
        updated_at: Set(Utc::now()),
        ..Default::default()
    }
    .insert(&tx)
    .await?;
    link(&tx, c.id, &n.alert_ids).await?;
    tx.commit().await?;
    audit(&st, &a, "case.create", "case", c.id, json!({"alert_ids": n.alert_ids})).await;
    Ok((StatusCode::CREATED, Json(json!({"case": c, "alert_ids": n.alert_ids}))))
}

#[derive(Deserialize)]
pub struct ListQ {
    status: Option<String>,
    limit: Option<u64>,
    offset: Option<u64>,
}

pub async fn list(State(st): State<AppState>, Query(q): Query<ListQ>) -> ApiResult<Json<Vec<case::Model>>> {
    let (limit, offset) = page(q.limit, q.offset);
    let mut s = case::Entity::find();
    if let Some(x) = q.status {
        s = s.filter(case::Column::Status.eq(x));
    }
    Ok(Json(s.order_by_desc(case::Column::CreatedAt).limit(limit).offset(offset).all(&st.db).await?))
}

pub async fn get_one(State(st): State<AppState>, Path(id): Path<i64>) -> ApiResult<Json<Value>> {
    let c = load(&st, id).await?;
    let alerts = alert::Entity::find().filter(alert::Column::CaseId.eq(id)).order_by_desc(alert::Column::Score).all(&st.db).await?;
    let notes = case_note::Entity::find().filter(case_note::Column::CaseId.eq(id)).order_by_asc(case_note::Column::CreatedAt).all(&st.db).await?;
    Ok(Json(json!({"case": c, "alerts": alerts, "notes": notes})))
}

#[derive(Deserialize)]
pub struct Patch {
    priority: Option<String>,
    assigned_to: Option<String>,
    status: Option<String>,
}

pub async fn update(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(id): Path<i64>, Json(p): Json<Patch>) -> ApiResult<Json<case::Model>> {
    a.require_triage()?;
    let c = load(&st, id).await?;
    if c.status == "closed" {
        return Err(ApiError::Conflict("case is closed".into()));
    }
    let mut am: case::ActiveModel = c.into();
    if let Some(pr) = p.priority {
        if !PRIORITIES.contains(&pr.as_str()) {
            return Err(ApiError::Unprocessable(vec![format!("priority must be one of {PRIORITIES:?}")]));
        }
        am.priority = Set(pr);
    }
    if let Some(s) = p.status {
        if !["open", "investigating"].contains(&s.as_str()) {
            return Err(ApiError::Unprocessable(vec!["status may only be set to open|investigating; use /close to close".into()]));
        }
        am.status = Set(s);
    }
    if let Some(x) = p.assigned_to {
        am.assigned_to = Set(Some(x));
    }
    am.updated_at = Set(Utc::now());
    let u = am.update(&st.db).await?;
    audit(&st, &a, "case.update", "case", id, json!({"status": u.status, "priority": u.priority})).await;
    Ok(Json(u))
}

#[derive(Deserialize)]
pub struct NoteReq {
    body: String,
}

pub async fn add_note(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(id): Path<i64>, Json(n): Json<NoteReq>) -> ApiResult<(StatusCode, Json<case_note::Model>)> {
    a.require_triage()?;
    load(&st, id).await?;
    if n.body.trim().is_empty() || n.body.len() > 10_000 {
        return Err(ApiError::Unprocessable(vec!["note body must be 1-10000 characters".into()]));
    }
    let m = case_note::ActiveModel { case_id: Set(id), author: Set(a.name.clone()), body: Set(n.body), created_at: Set(Utc::now()), ..Default::default() }
        .insert(&st.db)
        .await?;
    Ok((StatusCode::CREATED, Json(m)))
}

#[derive(Deserialize)]
pub struct AlertsReq {
    alert_ids: Vec<i64>,
}

pub async fn add_alerts(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(id): Path<i64>, Json(r): Json<AlertsReq>) -> ApiResult<Json<Value>> {
    a.require_triage()?;
    let c = load(&st, id).await?;
    if c.status == "closed" {
        return Err(ApiError::Conflict("case is closed".into()));
    }
    link(&st.db, id, &r.alert_ids).await?;
    audit(&st, &a, "case.add_alerts", "case", id, json!({"alert_ids": r.alert_ids})).await;
    Ok(Json(json!({"case_id": id, "linked": r.alert_ids})))
}

#[derive(Deserialize)]
pub struct CloseReq {
    /// `sar_filed` (admin / MLRO only) or `no_action`.
    outcome: String,
    sar_reference: Option<String>,
    note: String,
}

pub async fn close(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(id): Path<i64>, Json(r): Json<CloseReq>) -> ApiResult<Json<case::Model>> {
    a.require_triage()?;
    let c = load(&st, id).await?;
    if c.status == "closed" {
        return Err(ApiError::Conflict("case already closed".into()));
    }
    if r.note.trim().is_empty() {
        return Err(ApiError::Unprocessable(vec!["note is required to close a case".into()]));
    }
    let (alert_status, sar) = match r.outcome.as_str() {
        "sar_filed" => {
            a.require_admin()?;
            let s = r.sar_reference.as_deref().map(str::trim).filter(|s| !s.is_empty());
            match s {
                Some(s) => ("sar_filed", Some(s.to_string())),
                None => return Err(ApiError::Unprocessable(vec!["sar_reference is required when outcome is sar_filed".into()])),
            }
        }
        "no_action" => ("closed_false_positive", None),
        _ => return Err(ApiError::Unprocessable(vec!["outcome must be sar_filed or no_action".into()])),
    };
    let tx = st.db.begin().await?;
    let mut am: case::ActiveModel = c.into();
    am.status = Set("closed".into());
    am.outcome = Set(Some(r.outcome.clone()));
    am.sar_reference = Set(sar);
    am.closed_at = Set(Some(Utc::now()));
    am.updated_at = Set(Utc::now());
    let u = am.update(&tx).await?;
    alert::Entity::update_many()
        .col_expr(alert::Column::Status, Expr::value(alert_status))
        .col_expr(alert::Column::ResolutionNote, Expr::value(r.note.clone()))
        .col_expr(alert::Column::UpdatedAt, Expr::value(Utc::now()))
        .filter(alert::Column::CaseId.eq(id))
        .filter(alert::Column::Status.is_in(["open", "under_review", "escalated"]))
        .exec(&tx)
        .await?;
    case_note::ActiveModel { case_id: Set(id), author: Set(a.name.clone()), body: Set(format!("Case closed ({}): {}", r.outcome, r.note)), created_at: Set(Utc::now()), ..Default::default() }
        .insert(&tx)
        .await?;
    crate::feedback::capture_case(&tx, id, &a.name).await?;
    tx.commit().await?;
    audit(&st, &a, "case.close", "case", id, json!({"outcome": r.outcome, "sar_reference": u.sar_reference})).await;
    Ok(Json(u))
}
