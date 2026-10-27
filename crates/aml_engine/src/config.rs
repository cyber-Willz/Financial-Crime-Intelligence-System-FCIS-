use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const R_WATCHLIST: &str = "WATCHLIST_HIT";
pub const R_LARGE: &str = "LARGE_TXN";
pub const R_HRJ: &str = "HIGH_RISK_JURISDICTION";
pub const R_FX: &str = "CURRENCY_CONVERSION";
pub const R_FAN_IN: &str = "FAN_IN";
pub const R_FAN_OUT: &str = "FAN_OUT";
pub const R_HUB: &str = "STRUCTURING_HUB";
pub const R_PASS: &str = "PASS_THROUGH";
pub const R_LAYER: &str = "LAYERING_CHAIN";
pub const R_CYCLE: &str = "CIRCULAR_FLOW";
pub const R_PAIR: &str = "REPEATED_PAIR";
pub const R_BEHAVIOUR: &str = "BEHAVIOUR_CHANGE";
pub const R_DORMANT: &str = "DORMANT_REACTIVATION";
pub const R_CASH: &str = "CASH_VELOCITY";

pub const R_CUST_RISK: &str = "HIGH_RISK_CUSTOMER";
pub const R_CUST_JUR: &str = "CUSTOMER_JURISDICTION";
pub const R_INCOME: &str = "INCOME_MISMATCH";

pub const R_SANCTIONED: &str = "SANCTIONED_JURISDICTION";
pub const R_BAND: &str = "STRUCTURING_BAND";
pub const R_TRADE: &str = "TRADE_MISPRICING";
pub const R_DROP: &str = "AMOUNT_DROP";

pub const ALL_RULES: [&str; 21] = [
    R_WATCHLIST, R_LARGE, R_HRJ, R_FX, R_FAN_IN, R_FAN_OUT, R_HUB, R_PASS, R_LAYER, R_CYCLE,
    R_PAIR, R_BEHAVIOUR, R_DORMANT, R_CASH, R_CUST_RISK, R_CUST_JUR, R_INCOME,
    R_SANCTIONED, R_BAND, R_TRADE, R_DROP,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCfg {
    pub enabled: bool,
    /// Contribution to the noisy-OR risk score, 0..1.
    pub weight: f64,
    #[serde(default)]
    pub params: BTreeMap<String, f64>,
    #[serde(default)]
    pub description: String,
}

impl RuleCfg {
    pub fn p(&self, key: &str, default: f64) -> f64 {
        self.params.get(key).copied().unwrap_or(default)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Transactions scoring at or above this raise an alert.
    pub alert_threshold: u8,
    pub rules: BTreeMap<String, RuleCfg>,
    /// Comprehensively sanctioned jurisdictions: any payment touching them is escalated.
    #[serde(default)]
    pub sanctioned_countries: Vec<String>,
    #[serde(default)]
    pub very_high_risk_countries: Vec<String>,
    pub high_risk_countries: Vec<String>,
    pub medium_risk_countries: Vec<String>,
    /// Placeholder FX table (units of GBP per 1 unit of currency). Feed real rates in production.
    pub fx_to_gbp: BTreeMap<String, f64>,
}

impl EngineConfig {
    pub fn rule(&self, code: &str) -> RuleCfg {
        self.rules
            .get(code)
            .cloned()
            .unwrap_or_else(|| default_config().rules.remove(code).expect("known rule code"))
    }
    pub fn to_gbp(&self, amount: f64, currency: &str) -> f64 {
        amount * self.fx_to_gbp.get(&currency.to_ascii_uppercase()).copied().unwrap_or(1.0)
    }
}

fn rule(desc: &str, weight: f64, params: &[(&str, f64)]) -> RuleCfg {
    RuleCfg {
        enabled: true,
        weight,
        params: params.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        description: desc.to_string(),
    }
}

pub fn default_config() -> EngineConfig {
    let mut rules = BTreeMap::new();
    let mut add = |code: &str, r: RuleCfg| {
        rules.insert(code.to_string(), r);
    };
    add(R_WATCHLIST, rule("Sender or receiver account is on an internal watchlist / sanctions list", 0.95, &[]));
    add(R_LARGE, rule("Single transaction at or above the large-value threshold (GBP)", 0.45,
        &[("threshold", 100_000.0), ("extreme", 1_000_000.0), ("extreme_weight", 0.90), ("min_vs_mean", 0.0), ("min_history", 8.0)]));
    add(R_HRJ, rule("Transaction touches a high-risk (or very-high-risk) jurisdiction", 0.2,
        &[("medium_weight", 0.10), ("very_high_weight", 0.40)]));
    add(R_FX, rule("Currency conversion on the payment (payment vs received currency differ)", 0.05, &[]));
    add(R_FAN_IN, rule("Many distinct senders into one account in a short window", 0.5,
        &[("window_days", 7.0), ("min_counterparties", 6.0), ("min_focus", 0.5)]));
    add(R_FAN_OUT, rule("One account paying many distinct receivers in a short window", 0.35,
        &[("window_days", 7.0), ("min_counterparties", 6.0), ("min_focus", 0.5)]));
    add(R_HUB, rule("Slow fan-in of sub-threshold amounts from many senders (structuring hub)", 0.65,
        &[("window_days", 180.0), ("min_counterparties", 12.0), ("amount_min", 1_000.0), ("amount_max", 10_000.0), ("min_focus", 0.5)]));
    add(R_PASS, rule("Funds received and forwarded almost in full within hours (pass-through / rapid movement)", 0.3,
        &[("window_hours", 48.0), ("min_ratio", 0.75), ("max_ratio", 1.30), ("min_amount", 500.0)]));
    add(R_LAYER, rule("Chain of consecutive pass-through hops (layering)", 0.8, &[("min_depth", 3.0)]));
    add(R_CYCLE, rule("Funds return to the originator through a loop of accounts (round-tripping)", 0.8,
        &[("window_days", 14.0), ("min_hops", 3.0), ("max_hops", 5.0), ("amount_tol", 0.0), ("repeat_tol", 0.0)]));
    add(R_PAIR, rule("Same sender/receiver pair transacting repeatedly (smurfing / drip transfers)", 0.45,
        &[("window_days", 180.0), ("min_count", 8.0), ("cash_only", 1.0)]));
    add(R_BEHAVIOUR, rule("Amount far outside the sender's own historical pattern", 0.22,
        &[("min_history", 8.0), ("z_score", 4.0), ("ratio", 5.0)]));
    add(R_DORMANT, rule("Long-dormant account suddenly moves a material amount", 0.4,
        &[("dormant_days", 60.0), ("min_amount", 5_000.0)]));
    add(R_CASH, rule("High count of cash deposits/withdrawals on one account", 0.25,
        &[("window_days", 90.0), ("min_count", 10.0)]));

    add(R_CUST_RISK, rule("Sender or receiver belongs to a customer rated high risk (or a PEP)", 0.36, &[]));
    add(R_CUST_JUR, rule("High-risk customer transacting through a very-high-risk jurisdiction (combination is far worse than either alone)", 0.55, &[]));
    add(R_INCOME, rule("Amount or 30-day outflow out of proportion to the customer's declared income", 0.30,
        &[("single_multiple", 0.5), ("outflow_multiple", 1.0), ("window_days", 30.0)]));

    add(R_SANCTIONED, rule("Payment involving a sanctioned jurisdiction (OFAC-style screen). Counterparty side alerts alone; the sender's own country is weighted by origin_factor", 0.75,
        &[("origin_factor", 0.5)]));
    add(R_BAND, rule("Amount just below the reporting threshold; stronger when repeated by the same sender", 0.30,
        &[("threshold", 10_000.0), ("band_pct", 0.10), ("repeat_window_days", 7.0), ("repeat_min", 2.0), ("repeat_weight", 0.45),
         ("min_vs_mean", 1.5), ("min_history", 5.0), ("require_history", 1.0)]));
    add(R_TRADE, rule("Large payment from a trade-based customer (import/export): trade-based money laundering / mispricing", 0.55,
        &[("min_amount", 50_000.0)]));

    // Off by default: it is a scale-free signal that must be switched on per portfolio (see profiles/amlsim_fitted.json).
    let mut drop = rule("Payment far BELOW the sender's own usual amount (probing / layered micro-payments / structured small pieces)", 0.25,
        &[("min_history", 8.0), ("ratio", 0.1)]);
    drop.enabled = false;
    add(R_DROP, drop);

    let fx = [
        ("GBP", 1.0), ("USD", 0.79), ("EUR", 0.86), ("CHF", 0.90), ("JPY", 0.0053), ("INR", 0.0095),
        ("MXN", 0.043), ("TRY", 0.024), ("MAD", 0.080), ("AED", 0.215), ("ALL", 0.0088),
        ("PKR", 0.0028), ("NGN", 0.00052), ("BZD", 0.39),
    ];
    EngineConfig {
        alert_threshold: 60,
        rules,
        sanctioned_countries: ["SY", "RU", "SD", "IR", "KP", "VE"].iter().map(|s| s.to_string()).collect(),
        very_high_risk_countries: ["AF", "MM"].iter().map(|s| s.to_string()).collect(),
        high_risk_countries: ["NG", "MA", "AL", "AE", "MX", "TR", "PK"].iter().map(|s| s.to_string()).collect(),
        medium_risk_countries: vec![],
        fx_to_gbp: fx.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    }
}
