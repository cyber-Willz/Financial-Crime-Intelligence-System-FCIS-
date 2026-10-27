//! Stream a supported dataset through the same score+persist pipeline the API uses.
use crate::{app::AppState, ingest::{self, TxnRequest}, routes::customers};
use aml_datasets::{Format, Item};
use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

fn to_request(r: aml_datasets::Row) -> TxnRequest {
    TxnRequest {
        occurred_at: chrono::DateTime::from_timestamp(r.ts, 0),
        sender_account: r.sender,
        receiver_account: r.receiver,
        amount: r.amount,
        payment_currency: r.pcur.into(),
        received_currency: Some(r.rcur.into()),
        sender_country: r.scountry.as_str().into(),
        receiver_country: r.rcountry.as_str().into(),
        payment_type: r.ptype.as_str().into(),
        amount_gbp: Some(r.amount_gbp),
        label_laundering: r.label,
        label_typology: r.typology,
    }
}

#[derive(Default)]
struct Tally {
    done: usize,
    flagged: usize,
    alerts: usize,
}

async fn flush(st: &AppState, buf: &mut Vec<TxnRequest>, t: &mut Tally) -> Result<()> {
    if buf.is_empty() {
        return Ok(());
    }
    let d = ingest::ingest(st, std::mem::take(buf)).await.map_err(|e| anyhow::anyhow!("{e:?}"))?;
    t.done += d.len();
    t.flagged += d.iter().filter(|x| x.flagged).count();
    t.alerts += d.iter().filter(|x| x.alert_id.is_some()).count();
    Ok(())
}

pub async fn import_dataset(st: &AppState, file: &Path, fmt: Format, skip: usize, limit: Option<usize>, batch: usize) -> Result<()> {
    let t0 = Instant::now();
    let (mut seen, mut bad, mut customers_n) = (0usize, 0usize, 0usize);
    let mut pending_customers = vec![];
    let mut buf: Vec<TxnRequest> = Vec::with_capacity(batch);
    let mut tally = Tally::default();
    for item in aml_datasets::open(file, fmt, limit.map(|l| l + skip)).with_context(|| format!("opening {}", file.display()))? {
        match item {
            Item::Customer(c) => pending_customers.push(c),
            Item::Bad => bad += 1,
            Item::Row(r) => {
                if !pending_customers.is_empty() {
                    customers_n += customers::bulk_import(st, &pending_customers).await.map_err(|e| anyhow::anyhow!("{e:?}"))?;
                    pending_customers.clear();
                }
                seen += 1;
                if seen <= skip {
                    continue;
                }
                if r.sender == r.receiver {
                    bad += 1;
                    continue;
                }
                buf.push(to_request(r));
                if buf.len() >= batch {
                    flush(st, &mut buf, &mut tally).await?;
                    if (tally.done / batch) % 10 == 0 {
                        tracing::info!(imported = tally.done, flagged = tally.flagged, secs = t0.elapsed().as_secs(), "import progress");
                    }
                }
            }
        }
    }
    flush(st, &mut buf, &mut tally).await?;
    tracing::info!(imported = tally.done, customers = customers_n, skipped_invalid = bad, flagged_at_ingest = tally.flagged, alerts_raised = tally.alerts, secs = t0.elapsed().as_secs(), "import complete");
    Ok(())
}
