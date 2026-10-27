use super::{audit, page};
use crate::{app::AppState, auth::Actor, error::{ApiError, ApiResult}};
use aml_entity::{alert, transaction as t};
use axum::{extract::{Path, Query, State}, Extension, Json};
use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, Condition, EntityTrait, PaginatorTrait, QueryFilter, QueryOrder, QuerySelect, Set,
};
use serde::Deserialize;
use serde_json::{json, Value};

pub const STATUSES: [&str; 6] = ["open", "under_review", "escalated", "closed_false_positive", "closed_confirmed", "sar_filed"];

/// Alert workflow. `admin` may reopen closed alerts for review.
pub fn transition_allowed(from: &str, to: &str, admin: bool) -> bool {
    if from == to {
        return true;
    }
    match (from, to) {
        ("open", "under_review") | ("open", "closed_false_positive") => true,
        ("under_review", "escalated" | "closed_false_positive" | "closed_confirmed" | "open") => true,
        ("escalated", "sar_filed" | "closed_confirmed" | "closed_false_positive" | "under_review") => true,
        ("closed_false_positive" | "closed_confirmed", "under_review") => admin,
        _ => false,
    }
}

#[derive(Deserialize)]
pub struct Q {
    status: Option<String>,
    severity: Option<String>,
    account: Option<i64>,
    min_score: Option<i16>,
    limit: Option<u64>,
    offset: Option<u64>,
}

pub async fn list(State(st): State<AppState>, Query(q): Query<Q>) -> ApiResult<Json<Value>> {
    let (limit, offset) = page(q.limit, q.offset);
    let mut c = Condition::all();
    if let Some(s) = q.status {
        c = c.add(alert::Column::Status.eq(s));
    }
    if let Some(s) = q.severity {
        c = c.add(alert::Column::Severity.eq(s));
    }
    if let Some(a) = q.account {
        c = c.add(alert::Column::SubjectAccount.eq(a));
    }
    if let Some(m) = q.min_score {
        c = c.add(alert::Column::Score.gte(m));
    }
    let total = alert::Entity::find().filter(c.clone()).count(&st.db).await?;
    let items = alert::Entity::find()
        .filter(c)
        .order_by_desc(alert::Column::Score)
        .order_by_desc(alert::Column::Id)
        .limit(limit)
        .offset(offset)
        .all(&st.db)
        .await?;
    Ok(Json(json!({"total": total, "items": items})))
}

pub async fn get_one(State(st): State<AppState>, Path(id): Path<i64>) -> ApiResult<Json<Value>> {
    let a = alert::Entity::find_by_id(id).one(&st.db).await?.ok_or_else(|| ApiError::NotFound(format!("alert {id}")))?;
    let txns = t::Entity::find()
        .filter(
            Condition::all()
                .add(t::Column::Id.gte(a.first_txn_id))
                .add(t::Column::Id.lte(a.last_txn_id))
                .add(Condition::any().add(t::Column::SenderAccount.eq(a.subject_account)).add(t::Column::ReceiverAccount.eq(a.subject_account)))
                .add(t::Column::Flagged.eq(true)),
        )
        .order_by_asc(t::Column::OccurredAt)
        .limit(200)
        .all(&st.db)
        .await?;
    Ok(Json(json!({"alert": a, "transactions": txns})))
}

#[derive(Deserialize)]
pub struct Patch {
    status: Option<String>,
    assigned_to: Option<String>,
    resolution_note: Option<String>,
}

fn p_status_changed(a: &str, b: &str) -> bool {
    a != b
}

pub async fn update(State(st): State<AppState>, Extension(actor): Extension<Actor>, Path(id): Path<i64>, Json(p): Json<Patch>) -> ApiResult<Json<alert::Model>> {
    actor.require_triage()?;
    let cur = alert::Entity::find_by_id(id).one(&st.db).await?.ok_or_else(|| ApiError::NotFound(format!("alert {id}")))?;
    let mut am: alert::ActiveModel = cur.clone().into();
    if let Some(to) = &p.status {
        if !STATUSES.contains(&to.as_str()) {
            return Err(ApiError::Unprocessable(vec![format!("status must be one of {STATUSES:?}")]));
        }
        if !transition_allowed(&cur.status, to, actor.role == crate::auth::Role::Admin) {
            return Err(ApiError::Conflict(format!("cannot move alert from {} to {}", cur.status, to)));
        }
        if to == "sar_filed" {
            actor.require_admin()?;
        }
        let terminal = to.starts_with("closed") || to == "sar_filed";
        let note = p.resolution_note.as_deref().or(cur.resolution_note.as_deref()).unwrap_or("");
        if terminal && note.trim().is_empty() {
            return Err(ApiError::Unprocessable(vec!["resolution_note is required when closing an alert".into()]));
        }
        am.status = Set(to.clone());
    }
    if let Some(n) = p.resolution_note {
        am.resolution_note = Set(Some(n));
    }
    if let Some(a) = p.assigned_to {
        am.assigned_to = Set(Some(a));
    }
    am.updated_at = Set(Utc::now());
    let upd = am.update(&st.db).await?;
    if p_status_changed(&cur.status, &upd.status) {
        if upd.status.starts_with("closed") || upd.status == "sar_filed" {
            crate::feedback::capture_alert(&st.db, id, &actor.name).await?;
        } else if cur.status.starts_with("closed") || cur.status == "sar_filed" {
            crate::feedback::retract_alert(&st.db, id).await?;
        }
    }
    audit(&st, &actor, "alert.update", "alert", id, json!({"from": cur.status, "to": upd.status, "assigned_to": upd.assigned_to})).await;
    Ok(Json(upd))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn workflow_rules() {
        assert!(transition_allowed("open", "under_review", false));
        assert!(!transition_allowed("open", "sar_filed", true));
        assert!(!transition_allowed("closed_confirmed", "under_review", false));
        assert!(transition_allowed("closed_confirmed", "under_review", true));
        assert!(transition_allowed("escalated", "sar_filed", false));
        assert!(!transition_allowed("sar_filed", "open", true));
    }
}
