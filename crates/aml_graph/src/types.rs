//! Compact transaction record used by the graph layer.

use aml_engine::{Country, PaymentType};

/// One transaction as seen by the graph model.
///
/// `rule` is the rule engine's noisy-OR score in `[0, 1]` for this transaction (0 when unknown or
/// when the model is run without rule features). `label`/`typology` are only used for training and
/// evaluation; they are never read at inference time.
#[derive(Debug, Clone)]
pub struct GTxn {
    pub ts: i64,
    pub sender: i64,
    pub receiver: i64,
    pub amount: f64,
    pub ptype: PaymentType,
    pub sc: Country,
    pub rc: Country,
    pub cross_ccy: bool,
    /// Caller reference echoed back with scores (row number, database id, ...).
    pub rf: u32,
    pub label: bool,
    /// 0 = normal, >0 = index into the typology name table.
    pub typology: u8,
    /// causal engine score at processing time, in `[0, 1]` (the only rule signal the model may use)
    pub rule: f32,
    /// engine score after retroactive pattern linking (uses later transactions; baseline only, never a feature)
    pub rule_retro: f32,
}

impl GTxn {
    pub fn from_row(r: &aml_datasets::Row, rf: u32) -> Self {
        GTxn {
            ts: r.ts,
            sender: r.sender,
            receiver: r.receiver,
            amount: r.amount_gbp,
            ptype: r.ptype,
            sc: r.scountry,
            rc: r.rcountry,
            cross_ccy: r.pcur != r.rcur,
            rf,
            label: r.label.unwrap_or(false),
            typology: 0,
            rule: 0.0,
            rule_retro: 0.0,
        }
    }

    pub fn from_engine(t: &aml_engine::TxnIn, rf: u32) -> Self {
        GTxn {
            ts: t.ts,
            sender: t.sender,
            receiver: t.receiver,
            amount: t.amount_gbp,
            ptype: t.payment_type,
            sc: t.sender_country,
            rc: t.receiver_country,
            cross_ccy: t.cross_currency,
            rf,
            label: false,
            typology: 0,
            rule: 0.0,
            rule_retro: 0.0,
        }
    }
}
