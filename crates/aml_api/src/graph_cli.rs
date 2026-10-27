//! Graph layer batch jobs: `graph-rescore` writes per-account graph risk, `tune-graph` closes the feedback loop
//! by fine-tuning the model head on analyst outcomes (guarded: kept only if a held-out period improves).

use crate::ingest;
use aml_engine::{Country, PaymentType, TxnIn};
use aml_feedback::{tune::apply_outcomes, tune_head, TuneConfig};
use aml_graph::rules::score_rules;
use aml_graph::{CirGnn, GTxn, GraphScorer, RiskTable};
use anyhow::{bail, Context, Result};
use futures_util::TryStreamExt;
use sea_orm::{sqlx::Row, DatabaseConnection};
use std::collections::{HashMap, HashSet};
use std::path::Path;

async fn load_txns(db: &DatabaseConnection, days: i64) -> Result<Vec<GTxn>> {
    let pool = db.get_postgres_connection_pool();
    let mut rows = sea_orm::sqlx::query(
        "SELECT id, EXTRACT(EPOCH FROM occurred_at)::bigint AS ts, sender_account, receiver_account, amount_gbp::float8 AS gbp, \
                payment_type, sender_country, receiver_country, (payment_currency <> received_currency) AS xc \
           FROM transactions WHERE occurred_at >= COALESCE((SELECT max(occurred_at) FROM transactions), now()) - make_interval(days => $1::int) \
          ORDER BY occurred_at, id",
    )
    .bind(days as i32)
    .fetch(pool);
    let mut out = Vec::new();
    while let Some(r) = rows.try_next().await? {
        let (Some(pt), Some(sc), Some(rc)) = (PaymentType::parse(&r.get::<String, _>("payment_type")), Country::parse(&r.get::<String, _>("sender_country")), Country::parse(&r.get::<String, _>("receiver_country"))) else { continue };
        let id: i64 = r.get("id");
        let t = TxnIn { ts: r.get("ts"), sender: r.get("sender_account"), receiver: r.get("receiver_account"), amount_gbp: r.get("gbp"), payment_type: pt, sender_country: sc, receiver_country: rc, cross_currency: r.get("xc") };
        out.push(GTxn::from_engine(&t, id as u32));
    }
    Ok(out)
}

/// Stream the last `days` of transactions through the graph model and store each account's latest risk.
pub async fn rescore(db: &DatabaseConnection, model_path: &Path, days: i64) -> Result<()> {
    let model = CirGnn::load(model_path).map_err(|e| anyhow::anyhow!("{e}")).context("loading graph model")?;
    let wsecs = model.cfg.window_secs;
    let cfg = ingest::load_config(db).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut scorer = GraphScorer::new(model, &cfg);
    let txns = load_txns(db, days).await?;
    if txns.is_empty() {
        bail!("no transactions in the last {days} day(s)");
    }
    tracing::info!(transactions = txns.len(), "rescoring with the graph model");
    let mut latest: HashMap<i64, (f32, i64)> = HashMap::new();
    let mut take = |w: aml_graph::WindowScores| {
        for (a, r) in w.accounts {
            latest.insert(a, (r, w.window * wsecs));
        }
    };
    for t in txns {
        match scorer.push(t) {
            Ok(Some(w)) => take(w),
            Ok(None) => {}
            Err(_) => {}
        }
    }
    if let Some(w) = scorer.flush() {
        take(w);
    }
    let name = model_path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let all: Vec<(i64, f32, i64)> = latest.into_iter().map(|(a, (r, w))| (a, r.clamp(0.0, 1.0), w)).collect();
    let pool = db.get_postgres_connection_pool();
    for chunk in all.chunks(5000) {
        let accts: Vec<i64> = chunk.iter().map(|x| x.0).collect();
        let risks: Vec<f32> = chunk.iter().map(|x| x.1).collect();
        let wins: Vec<f64> = chunk.iter().map(|x| x.2 as f64).collect();
        sea_orm::sqlx::query(
            "INSERT INTO graph_risk (account, risk, window_start, model) SELECT a, r, to_timestamp(w), $4 FROM UNNEST($1::bigint[], $2::real[], $3::float8[]) AS u(a, r, w) \
             ON CONFLICT (account) DO UPDATE SET risk = EXCLUDED.risk, window_start = EXCLUDED.window_start, model = EXCLUDED.model, scored_at = now()",
        )
        .bind(&accts)
        .bind(&risks)
        .bind(&wins)
        .bind(&name)
        .execute(pool)
        .await?;
    }
    tracing::info!(accounts = all.len(), "graph risk stored");
    Ok(())
}

/// Fine-tune the model head on analyst outcomes; writes `out` only when the held-out period improves.
pub async fn tune(db: &DatabaseConnection, model_path: &Path, out: &Path, days: i64) -> Result<()> {
    let mut model = CirGnn::load(model_path).map_err(|e| anyhow::anyhow!("{e}")).context("loading graph model")?;
    let cfg = ingest::load_config(db).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut txns = load_txns(db, days).await?;
    // confirmed / SAR transactions of reviewed alerts are positives; everything else is negative
    let pool = db.get_postgres_connection_pool();
    let rows = sea_orm::sqlx::query(
        "SELECT t.id FROM feedback_events f JOIN alerts a ON a.id = f.alert_id \
           JOIN transactions t ON t.id BETWEEN a.first_txn_id AND a.last_txn_id AND (t.sender_account = a.subject_account OR t.receiver_account = a.subject_account) AND t.flagged \
          WHERE f.decision IN ('sar_filed','confirmed')",
    )
    .fetch_all(pool)
    .await?;
    let pos: HashSet<u32> = rows.iter().map(|r| r.get::<i64, _>(0) as u32).collect();
    let outcomes: HashMap<u32, bool> = pos.into_iter().map(|i| (i, true)).collect();
    let n = apply_outcomes(&mut txns, &outcomes);
    tracing::info!(transactions = txns.len(), confirmed = n, "analyst outcomes applied as labels");
    score_rules(&mut txns, &cfg);
    let risk = RiskTable::from_engine(&cfg);
    let rep = tune_head(&mut model, &txns, &risk, &TuneConfig::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", serde_json::to_string_pretty(&rep)?);
    if rep.accepted {
        model.save(out).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("tuned model written to {}", out.display());
    } else {
        println!("model not changed: {}", rep.reason);
    }
    Ok(())
}
