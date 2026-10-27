use crate::{app::AppState, error::ApiResult};
use axum::{extract::State, Json};
use sea_orm::sqlx::{self, Row};
use serde_json::{json, Value};

pub async fn overview(State(st): State<AppState>) -> ApiResult<Json<Value>> {
    let pool = st.db.get_postgres_connection_pool();
    let tot = sqlx::query(
        "SELECT (SELECT count(*) FROM transactions) AS txns, (SELECT count(*) FROM transactions WHERE flagged) AS flagged, \
                (SELECT count(*) FROM accounts) AS accounts, (SELECT count(*) FROM accounts WHERE watchlisted) AS watchlisted, \
                (SELECT count(*) FROM cases WHERE status <> 'closed') AS open_cases",
    )
    .fetch_one(pool)
    .await?;
    let group = |rows: Vec<sqlx::postgres::PgRow>| -> Value {
        Value::Object(rows.iter().map(|r| (r.get::<String, _>("k"), json!(r.get::<i64, _>("n")))).collect())
    };
    let by_status = group(sqlx::query("SELECT status AS k, count(*) AS n FROM alerts GROUP BY 1").fetch_all(pool).await?);
    let by_sev = group(sqlx::query("SELECT severity AS k, count(*) AS n FROM alerts GROUP BY 1").fetch_all(pool).await?);
    let by_risk = group(sqlx::query("SELECT risk_level AS k, count(*) AS n FROM accounts GROUP BY 1").fetch_all(pool).await?);
    let rules = group(
        sqlx::query("SELECT x AS k, count(*) AS n FROM alerts, jsonb_array_elements_text(rule_codes) x GROUP BY 1 ORDER BY 2 DESC LIMIT 15")
            .fetch_all(pool)
            .await?,
    );
    let top = sqlx::query("SELECT account_number, risk_score, risk_level, alert_count FROM accounts ORDER BY risk_score DESC, alert_count DESC LIMIT 10")
        .fetch_all(pool)
        .await?;
    let top: Vec<Value> = top
        .iter()
        .map(|r| json!({"account": r.get::<i64, _>("account_number"), "risk_score": r.get::<i16, _>("risk_score"), "risk_level": r.get::<String, _>("risk_level"), "alert_count": r.get::<i32, _>("alert_count")}))
        .collect();
    Ok(Json(json!({
        "transactions": tot.get::<i64, _>("txns"),
        "flagged_transactions": tot.get::<i64, _>("flagged"),
        "accounts": tot.get::<i64, _>("accounts"),
        "watchlisted_accounts": tot.get::<i64, _>("watchlisted"),
        "open_cases": tot.get::<i64, _>("open_cases"),
        "alerts_by_status": by_status,
        "alerts_by_severity": by_sev,
        "accounts_by_risk_level": by_risk,
        "alerts_by_rule": rules,
        "top_risk_accounts": top,
    })))
}
