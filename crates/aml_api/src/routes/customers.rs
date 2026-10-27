//! Customer due-diligence (KYC) profiles: risk rating, PEP flag, declared income, linked accounts.
//! The engine uses these for HIGH_RISK_CUSTOMER, CUSTOMER_JURISDICTION and INCOME_MISMATCH.
use super::{audit, page};
use crate::{app::AppState, auth::Actor, error::{ApiError, ApiResult}};
use aml_datasets::CustomerRow;
use aml_engine::{Country, CustomerRisk, Profile};
use aml_entity::{account, alert, customer};
use axum::{extract::{Path, Query, State}, Extension, Json};
use sea_orm::{
    sqlx::{self, Row},
    ColumnTrait, EntityTrait, QueryFilter, QueryOrder, QuerySelect,
};
use serde::Deserialize;
use serde_json::{json, Value};

const TYPES: [&str; 7] = ["individual", "business", "employee", "corporate", "financial_institution", "ngo", "other"];

#[derive(Deserialize)]
pub struct Upsert {
    customer_type: Option<String>,
    country: Option<String>,
    /// Business sector, e.g. "Import/Export". Trade-based sectors are monitored for trade mispricing.
    sector: Option<String>,
    /// low | medium | high
    risk_rating: String,
    #[serde(default)]
    pep: bool,
    annual_income_gbp: Option<f64>,
    /// Account numbers owned by this customer (linking is additive; an account can belong to one customer).
    #[serde(default)]
    accounts: Vec<i64>,
}

pub async fn put(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(cref): Path<String>, Json(u): Json<Upsert>) -> ApiResult<Json<Value>> {
    a.require_triage()?;
    let mut errs = vec![];
    if cref.is_empty() || cref.len() > 64 {
        errs.push("customer reference must be 1-64 characters".to_string());
    }
    let ctype = u.customer_type.clone().unwrap_or_else(|| "individual".into());
    if !TYPES.contains(&ctype.as_str()) {
        errs.push(format!("customer_type must be one of {TYPES:?}"));
    }
    let risk = CustomerRisk::parse(&u.risk_rating);
    if risk.is_none() {
        errs.push("risk_rating must be low|medium|high".into());
    }
    let country = match u.country.as_deref().map(|c| Country::parse(c).map(|c| c.as_str().to_string())) {
        Some(None) => {
            errs.push("country must be ISO-3166 alpha-2".into());
            None
        }
        Some(Some(c)) => Some(c),
        None => None,
    };
    if u.annual_income_gbp.is_some_and(|v| !v.is_finite() || v < 0.0) {
        errs.push("annual_income_gbp must be a non-negative number".into());
    }
    if u.accounts.len() > 50 || u.accounts.iter().any(|a| *a <= 0) {
        errs.push("accounts: at most 50 positive account numbers".into());
    }
    if !errs.is_empty() {
        return Err(ApiError::Unprocessable(errs));
    }
    let pool = st.db.get_postgres_connection_pool();
    let mut tx = pool.begin().await?;
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO customers (external_ref, customer_type, country, risk_rating, pep, annual_income_gbp, sector) \
         VALUES ($1,$2,$3,$4,$5,ROUND($6::float8::numeric,2),$7) \
         ON CONFLICT (external_ref) DO UPDATE SET customer_type=EXCLUDED.customer_type, country=EXCLUDED.country, sector=EXCLUDED.sector, \
            risk_rating=EXCLUDED.risk_rating, pep=EXCLUDED.pep, annual_income_gbp=EXCLUDED.annual_income_gbp, updated_at=now() \
         RETURNING id",
    )
    .bind(&cref).bind(&ctype).bind(&country).bind(risk.unwrap().as_str()).bind(u.pep).bind(u.annual_income_gbp).bind(&u.sector)
    .fetch_one(&mut *tx)
    .await?;
    if !u.accounts.is_empty() {
        sqlx::query("INSERT INTO accounts (account_number, country, first_seen, last_seen) SELECT x, $2, now(), now() FROM UNNEST($1::bigint[]) x ON CONFLICT DO NOTHING")
            .bind(&u.accounts).bind(&country).execute(&mut *tx).await?;
        let taken: Vec<i64> = sqlx::query_scalar("SELECT account_number FROM accounts WHERE account_number = ANY($1) AND customer_id IS NOT NULL AND customer_id <> $2")
            .bind(&u.accounts).bind(id).fetch_all(&mut *tx).await?;
        if !taken.is_empty() {
            return Err(ApiError::Conflict(format!("accounts already belong to another customer: {taken:?}")));
        }
        sqlx::query("UPDATE accounts SET customer_id = $2, updated_at = now() WHERE account_number = ANY($1)")
            .bind(&u.accounts).bind(id).execute(&mut *tx).await?;
    }
    let linked: Vec<i64> = sqlx::query_scalar("SELECT account_number FROM accounts WHERE customer_id = $1 ORDER BY 1").bind(id).fetch_all(&mut *tx).await?;
    tx.commit().await?;
    {
        let mut eng = st.engine.lock().await;
        let p = Profile {
            risk: risk.unwrap(),
            annual_income_gbp: u.annual_income_gbp,
            pep: u.pep,
            trade_exposed: u.sector.as_deref().map(aml_engine::is_trade_sector).unwrap_or(false),
        };
        for acct in &linked {
            eng.set_profile(*acct, &p);
        }
    }
    audit(&st, &a, "customer.upsert", "customer", &cref, json!({"risk_rating": u.risk_rating, "pep": u.pep, "accounts": linked})).await;
    Ok(Json(json!({"id": id, "external_ref": cref, "risk_rating": u.risk_rating, "pep": u.pep, "accounts": linked})))
}

#[derive(Deserialize)]
pub struct ListQ {
    risk_rating: Option<String>,
    limit: Option<u64>,
    offset: Option<u64>,
}

pub async fn list(State(st): State<AppState>, Query(q): Query<ListQ>) -> ApiResult<Json<Vec<customer::Model>>> {
    let (limit, offset) = page(q.limit, q.offset);
    let mut s = customer::Entity::find();
    if let Some(r) = q.risk_rating {
        s = s.filter(customer::Column::RiskRating.eq(r));
    }
    Ok(Json(s.order_by_desc(customer::Column::UpdatedAt).limit(limit).offset(offset).all(&st.db).await?))
}

pub async fn get_one(State(st): State<AppState>, Path(cref): Path<String>) -> ApiResult<Json<Value>> {
    let c = customer::Entity::find().filter(customer::Column::ExternalRef.eq(cref.clone())).one(&st.db).await?
        .ok_or_else(|| ApiError::NotFound(format!("customer {cref}")))?;
    let accounts = account::Entity::find().filter(account::Column::CustomerId.eq(c.id)).order_by_desc(account::Column::RiskScore).all(&st.db).await?;
    let ids: Vec<i64> = accounts.iter().map(|a| a.account_number).collect();
    let alerts = alert::Entity::find().filter(alert::Column::SubjectAccount.is_in(ids)).order_by_desc(alert::Column::Score).limit(25).all(&st.db).await?;
    Ok(Json(json!({"customer": c, "accounts": accounts, "alerts": alerts})))
}

/// Bulk upsert used by dataset import: customers, their accounts, the link, the engine profile, and the
/// watchlist entries for customers that are on a sanctions list.
pub async fn bulk_import(st: &AppState, rows: &[CustomerRow]) -> ApiResult<usize> {
    let pool = st.db.get_postgres_connection_pool();
    for chunk in rows.chunks(2000) {
        let mut tx = pool.begin().await?;
        let refs: Vec<String> = chunk.iter().map(|c| c.external_ref.clone()).collect();
        let ratings: Vec<String> = chunk.iter().map(|c| c.risk.as_str().to_string()).collect();
        let incomes: Vec<f64> = chunk.iter().map(|c| c.annual_income_gbp.unwrap_or(-1.0)).collect();
        let types: Vec<String> = chunk.iter().map(|c| c.customer_type.to_string()).collect();
        let sectors: Vec<String> = chunk.iter().map(|c| c.sector.clone().unwrap_or_default()).collect();
        let peps: Vec<bool> = chunk.iter().map(|c| c.pep).collect();
        let ids = sqlx::query(
            "INSERT INTO customers (external_ref, risk_rating, annual_income_gbp, customer_type, sector, pep) \
             SELECT r, k, CASE WHEN i < 0 THEN NULL ELSE ROUND(i::numeric,2) END, t, NULLIF(sc,''), p \
             FROM UNNEST($1::text[], $2::text[], $3::float8[], $4::text[], $5::text[], $6::bool[]) AS u(r,k,i,t,sc,p) \
             ON CONFLICT (external_ref) DO UPDATE SET risk_rating=EXCLUDED.risk_rating, annual_income_gbp=EXCLUDED.annual_income_gbp, \
                customer_type=EXCLUDED.customer_type, sector=EXCLUDED.sector, pep=EXCLUDED.pep, updated_at=now() \
             RETURNING id, external_ref",
        )
        .bind(&refs).bind(&ratings).bind(&incomes).bind(&types).bind(&sectors).bind(&peps)
        .fetch_all(&mut *tx)
        .await?;
        let by_ref: std::collections::HashMap<String, i64> = ids.iter().map(|r| (r.get::<String, _>("external_ref"), r.get::<i64, _>("id"))).collect();
        let accts: Vec<i64> = chunk.iter().map(|c| c.account).collect();
        let cids: Vec<i64> = chunk.iter().map(|c| by_ref[&c.external_ref]).collect();
        sqlx::query("INSERT INTO accounts (account_number, first_seen, last_seen) SELECT x, now(), now() FROM UNNEST($1::bigint[]) x ON CONFLICT DO NOTHING")
            .bind(&accts).execute(&mut *tx).await?;
        sqlx::query("UPDATE accounts SET customer_id = u.c, updated_at = now() FROM UNNEST($1::bigint[], $2::bigint[]) AS u(a,c) WHERE accounts.account_number = u.a")
            .bind(&accts).bind(&cids).execute(&mut *tx).await?;
        let sanctioned: Vec<i64> = chunk.iter().filter(|c| c.sanctioned).map(|c| c.account).collect();
        if !sanctioned.is_empty() {
            sqlx::query(
                "INSERT INTO watchlist (account_number, list_name, reason, added_by) \
                 SELECT x, 'SANCTIONS', 'customer flagged as sanctioned in imported KYC data', 'import' FROM UNNEST($1::bigint[]) x \
                 ON CONFLICT (account_number) DO NOTHING",
            )
            .bind(&sanctioned).execute(&mut *tx).await?;
            sqlx::query("UPDATE accounts SET watchlisted = TRUE WHERE account_number = ANY($1)").bind(&sanctioned).execute(&mut *tx).await?;
        }
        tx.commit().await?;
        let mut eng = st.engine.lock().await;
        for c in chunk {
            eng.set_profile(c.account, &Profile { risk: c.risk, annual_income_gbp: c.annual_income_gbp, pep: c.pep, trade_exposed: c.trade_exposed() });
            if c.sanctioned {
                eng.set_watch(c.account, true);
            }
        }
    }
    Ok(rows.len())
}
