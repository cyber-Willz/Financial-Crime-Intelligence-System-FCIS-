//! Adapters that map public AML / fraud datasets onto the engine's canonical transaction model.
//!
//! | format             | file                                   | what it provides                                   |
//! |--------------------|----------------------------------------|----------------------------------------------------|
//! | `saml-d`           | SAML-D.csv (9.5M, labelled typologies) | account-to-account graph, 17 laundering typologies |
//! | `aml-suspicious`   | AML_Suspicious_Transactions.csv        | customers with KYC rating + income, country, label |
//! | `risk-prediction`  | financial_transactions.csv             | sender/receiver accounts (false-positive control)  |
//! | `fraudshield`      | FraudShield_Banking_Data.csv           | customer -> merchant card flows (FP control)       |
//! | `kyc-risk`         | transactions_with_fatf_ofac.csv (+clients) | KYC attributes, OFAC/FATF/structuring/trade flags |
//! | `lending`          | loan_information_data.csv (+customers) | loan disbursements to customers with declared income |
//!
//! Account-number namespaces are disjoint per dataset (customers: 1e8 aml-suspicious, 2e8 kyc-risk, 3e8 lending,
//! 4e8 fraudshield; counterparties 9.0e9/9.1e9 per-country correspondents, 3e9 merchants, 4e9 lender), so several
//! datasets can be loaded into one database without their identities colliding.
//!
//! Datasets that are not time-ordered are sorted before being yielded, because the engine's rolling
//! windows assume chronological input. `Risk_Score` in AML_Suspicious is deliberately **never** read:
//! it perfectly determines the label (AUC 1.0), so using it would be target leakage.

use aml_engine::{saml, Country, CustomerRisk, PaymentType};
use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    SamlD,
    AmlSuspicious,
    RiskPrediction,
    FraudShield,
    KycRisk,
    Lending,
    /// IBM AMLSim-style export: transactions.csv (+ alerts.csv for typology names) with integer day-step timestamps.
    AmlSim,
}

impl Format {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "saml-d" | "samld" => Self::SamlD,
            "aml-suspicious" => Self::AmlSuspicious,
            "risk-prediction" => Self::RiskPrediction,
            "fraudshield" | "fraud-shield" => Self::FraudShield,
            "kyc-risk" | "kyc" => Self::KycRisk,
            "lending" => Self::Lending,
            "amlsim" | "aml-sim" => Self::AmlSim,
            other => bail!("unknown format '{other}' (saml-d | aml-suspicious | risk-prediction | fraudshield | kyc-risk | lending | amlsim)"),
        })
    }
    /// Datasets whose rows are not chronological, or that carry no time ordering guarantee.
    pub fn needs_sort(&self) -> bool {
        !matches!(self, Format::SamlD)
    }
    /// Sensible default warm-up (days) before metrics count.
    pub fn default_warmup_days(&self) -> i64 {
        match self {
            Format::SamlD | Format::KycRisk => 14,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Row {
    pub ts: i64,
    pub sender: i64,
    pub receiver: i64,
    pub amount: f64,
    pub amount_gbp: f64,
    pub pcur: &'static str,
    pub rcur: &'static str,
    pub scountry: Country,
    pub rcountry: Country,
    pub ptype: PaymentType,
    pub label: Option<bool>,
    pub typology: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CustomerRow {
    pub account: i64,
    pub external_ref: String,
    pub risk: CustomerRisk,
    pub annual_income_gbp: Option<f64>,
    /// individual | business | employee | corporate | financial_institution | ngo | other
    pub customer_type: &'static str,
    pub sector: Option<String>,
    pub pep: bool,
    /// On a sanctions list: the account is added to the watchlist.
    pub sanctioned: bool,
}

impl CustomerRow {
    /// Import/export (trade-based) sectors are monitored for trade mispricing.
    pub fn trade_exposed(&self) -> bool {
        self.sector.as_deref().map(aml_engine::is_trade_sector).unwrap_or(false)
    }
    pub fn basic(account: i64, external_ref: String, risk: CustomerRisk, income: Option<f64>) -> Self {
        CustomerRow { account, external_ref, risk, annual_income_gbp: income, customer_type: "individual", sector: None, pep: false, sanctioned: false }
    }
}

#[derive(Debug)]
pub enum Item {
    Customer(CustomerRow),
    Row(Row),
    /// A row that could not be mapped (counted by the caller, never silently dropped).
    Bad,
}

// Placeholder FX used only by adapters whose source has no GBP figure. Documented in DATASETS.md.
const USD_GBP: f64 = 0.79;
const PKR_GBP: f64 = 0.0028;

fn digits(s: &str) -> Option<i64> {
    let d: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    d.parse().ok()
}

/// "2024-08-27 12:10:36.928669", "2025-01-02T09:52:38Z", "2025-03-08" + "10:54" -> unix seconds (UTC).
pub fn parse_datetime(date: &str, time: Option<&str>) -> Option<i64> {
    let (d, t) = match time {
        Some(t) => (date.trim(), t.trim()),
        None => {
            let s = date.trim();
            let mut it = s.splitn(2, |c| c == ' ' || c == 'T');
            (it.next()?, it.next().unwrap_or("00:00:00"))
        }
    };
    let t = t.trim_end_matches('Z');
    let t = t.split('.').next().unwrap_or(t);
    saml::parse_ts(d, t)
}

fn country(code: &str) -> Country {
    Country::parse(code).expect("static country code")
}

fn city_country(city: &str) -> Option<&'static str> {
    Some(match city.trim() {
        "Faisalabad" | "Islamabad" | "Karachi" | "Lahore" | "Multan" => "PK",
        "Singapore" => "SG",
        "Dubai" => "AE",
        "London" => "GB",
        "Bangkok" => "TH",
        "Kuala Lumpur" => "MY",
        _ => return None,
    })
}

fn ordinal(c: Country) -> i64 {
    (c.0[0].wrapping_sub(b'A') as i64) * 26 + c.0[1].wrapping_sub(b'A') as i64
}

fn col(h: &csv::StringRecord, name: &str) -> Result<usize> {
    h.iter().position(|x| x.trim() == name).ok_or_else(|| anyhow!("missing column '{name}'"))
}

pub fn open(path: &Path, fmt: Format, limit: Option<usize>) -> Result<Box<dyn Iterator<Item = Item>>> {
    match fmt {
        Format::SamlD => open_saml(path, limit),
        Format::AmlSuspicious => sorted(aml_suspicious(path, limit)?),
        Format::RiskPrediction => sorted((vec![], risk_prediction(path, limit)?)),
        Format::FraudShield => sorted((vec![], fraudshield(path, limit)?)),
        Format::KycRisk => sorted(kyc_risk(path, limit)?),
        Format::Lending => sorted(lending(path, limit)?),
        Format::AmlSim => sorted((vec![], amlsim(path, limit)?)),
    }
}

fn sorted((customers, mut rows): (Vec<CustomerRow>, Vec<Option<Row>>)) -> Result<Box<dyn Iterator<Item = Item>>> {
    let (mut good, bad): (Vec<Row>, Vec<()>) = {
        let mut g = vec![];
        let mut b = vec![];
        for r in rows.drain(..) {
            match r {
                Some(r) => g.push(r),
                None => b.push(()),
            }
        }
        (g, b)
    };
    good.sort_by_key(|r| r.ts);
    Ok(Box::new(
        customers.into_iter().map(Item::Customer).chain(bad.into_iter().map(|_| Item::Bad)).chain(good.into_iter().map(Item::Row)),
    ))
}

fn reader(path: &Path) -> Result<csv::Reader<std::fs::File>> {
    csv::ReaderBuilder::new().flexible(true).trim(csv::Trim::All).from_path(path).with_context(|| format!("opening {}", path.display()))
}

// ------------------------------------------------------------------------------------ SAML-D
fn open_saml(path: &Path, limit: Option<usize>) -> Result<Box<dyn Iterator<Item = Item>>> {
    let rdr = csv::ReaderBuilder::new().has_headers(true).from_path(path).with_context(|| format!("opening {}", path.display()))?;
    let it = rdr.into_records().map(|rec| {
        let Ok(rec) = rec else { return Item::Bad };
        let g = |i: usize| rec.get(i).unwrap_or("");
        let parsed = (|| -> Option<Row> {
            let ts = saml::parse_ts(g(1), g(0))?;
            let amount: f64 = g(4).parse().ok()?;
            let laundering = g(10) == "1";
            Some(Row {
                ts,
                sender: g(2).parse().ok()?,
                receiver: g(3).parse().ok()?,
                amount,
                // SAML-D amounts are already on one comparable scale: no FX is applied.
                amount_gbp: amount,
                pcur: saml::currency_iso(g(5))?,
                rcur: saml::currency_iso(g(6))?,
                scountry: Country::parse(saml::country_iso2(g(7))?)?,
                rcountry: Country::parse(saml::country_iso2(g(8))?)?,
                ptype: PaymentType::parse(g(9))?,
                label: Some(laundering),
                typology: if laundering { Some(g(11).to_string()) } else { None },
            })
        })();
        match parsed {
            Some(r) if r.sender != r.receiver => Item::Row(r),
            _ => Item::Bad,
        }
    });
    Ok(match limit {
        Some(l) => Box::new(it.take(l)),
        None => Box::new(it),
    })
}

// ------------------------------------------------------------------------- AML_Suspicious
fn aml_suspicious(path: &Path, limit: Option<usize>) -> Result<(Vec<CustomerRow>, Vec<Option<Row>>)> {
    let mut rdr = reader(path)?;
    let h = rdr.headers()?.clone();
    let (ic, id, it, ico, ia, ip, ii, il) = (
        col(&h, "Customer_ID")?, col(&h, "Transaction_Date")?, col(&h, "Transaction_Type")?, col(&h, "Country")?,
        col(&h, "Amount")?, col(&h, "Customer_Risk_Profile")?, col(&h, "Annual_Income")?, col(&h, "Suspicious_Flag")?,
    );
    let mut customers: HashMap<i64, CustomerRow> = HashMap::new();
    let mut order = vec![];
    let mut rows = vec![];
    for rec in rdr.records().take(limit.unwrap_or(usize::MAX)) {
        let rec = rec?;
        let r = (|| -> Option<Row> {
            let acct = 100_000_000 + digits(&rec[ic])?;
            let c = Country::parse(saml::country_iso2(&rec[ico])?)?;
            let amount: f64 = rec[ia].parse().ok()?;
            let flag = &rec[il] == "1";
            let risk = CustomerRisk::parse(&rec[ip])?;
            let income: f64 = rec[ii].parse().ok()?;
            if !customers.contains_key(&acct) {
                customers.insert(acct, CustomerRow::basic(acct, rec[ic].to_string(), risk, Some(income)));
                order.push(acct);
            }
            Some(Row {
                ts: parse_datetime(&rec[id], None)?,
                sender: acct,
                // no counterparty in the source: one pseudo "correspondent" account per destination country
                receiver: 9_000_000_000 + ordinal(c),
                amount,
                amount_gbp: amount,
                pcur: "GBP",
                rcur: "GBP",
                scountry: c,
                rcountry: c,
                ptype: PaymentType::parse(&rec[it])?,
                label: Some(flag),
                typology: flag.then(|| "Suspicious_Flag".to_string()),
            })
        })();
        rows.push(r);
    }
    let customers = order.into_iter().filter_map(|a| customers.remove(&a)).collect();
    Ok((customers, rows))
}

// ----------------------------------------------------------------------- Risk_prediction
fn risk_prediction(path: &Path, limit: Option<usize>) -> Result<Vec<Option<Row>>> {
    let mut rdr = reader(path)?;
    let h = rdr.headers()?.clone();
    let (it, is, ir, ia, ity, il, iflag) = (
        col(&h, "Timestamp")?, col(&h, "Sender Account ID")?, col(&h, "Receiver Account ID")?, col(&h, "Amount")?,
        col(&h, "Type")?, col(&h, "Location")?, col(&h, "Suspicious Activity Flag")?,
    );
    let _ = il;
    let mut rows = vec![];
    for rec in rdr.records().take(limit.unwrap_or(usize::MAX)) {
        let rec = rec?;
        rows.push((|| -> Option<Row> {
            let amount: f64 = rec[ia].parse().ok()?;
            let flag = &rec[iflag] == "1";
            Some(Row {
                ts: parse_datetime(&rec[it], None)?,
                sender: 1_000_000_000 + digits(&rec[is])?,
                receiver: 2_000_000_000 + digits(&rec[ir])?,
                amount,
                amount_gbp: amount * USD_GBP,
                pcur: "USD",
                rcur: "USD",
                scountry: country("US"),
                rcountry: country("US"),
                ptype: match &rec[ity] {
                    "Deposit" => PaymentType::CashDeposit,
                    "Withdrawal" => PaymentType::CashWithdrawal,
                    "Transfer" => PaymentType::OnlineTransfer,
                    _ => return None,
                },
                label: Some(flag),
                typology: flag.then(|| "Suspicious_Activity_Flag".to_string()),
            })
        })());
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------------- AMLSim
/// AMLSim account ids are small integers; they are shifted into their own namespace.
pub const AMLSIM_BASE: i64 = 500_000_000;
const SECS_PER_STEP: i64 = 86_400;

/// `transactions.csv` columns: TX_ID, SENDER_ACCOUNT_ID, RECEIVER_ACCOUNT_ID, TX_TYPE, TX_AMOUNT, TIMESTAMP, IS_FRAUD, ALERT_ID.
/// One step = one day (the export has no wall-clock time). All accounts are US and every transfer is domestic, so the
/// only information in a row is who paid whom, when, and how much. `alerts.csv`, when next to the file, supplies the
/// typology name (`cycle` / `fan_in`) for labelled rows. Amounts are in an unknown unit and are treated as USD.
fn amlsim(path: &Path, limit: Option<usize>) -> Result<Vec<Option<Row>>> {
    let mut types: HashMap<i64, String> = HashMap::new();
    if let Ok(ap) = sibling(path, "alerts.csv") {
        let mut r = reader(&ap)?;
        let h = r.headers()?.clone();
        let (ia, it) = (col(&h, "ALERT_ID")?, col(&h, "ALERT_TYPE")?);
        for rec in r.records() {
            let rec = rec?;
            if let Ok(id) = rec[ia].parse::<i64>() {
                types.entry(id).or_insert_with(|| rec[it].to_string());
            }
        }
    }
    let mut rdr = reader(path)?;
    let h = rdr.headers()?.clone();
    let (is, ir, ia, itm, ifr, ial) = (col(&h, "SENDER_ACCOUNT_ID")?, col(&h, "RECEIVER_ACCOUNT_ID")?, col(&h, "TX_AMOUNT")?, col(&h, "TIMESTAMP")?, col(&h, "IS_FRAUD")?, col(&h, "ALERT_ID")?);
    let mut rows = vec![];
    for rec in rdr.records().take(limit.unwrap_or(usize::MAX)) {
        let rec = rec?;
        rows.push((|| -> Option<Row> {
            let amount = rec[ia].parse::<f64>().ok().filter(|a| a.is_finite() && *a > 0.0)?; // zero-amount rows are data errors: counted as unmappable
            let flag = rec[ifr].eq_ignore_ascii_case("true");
            let alert = rec[ial].parse::<i64>().unwrap_or(-1);
            Some(Row {
                ts: rec[itm].parse::<i64>().ok()? * SECS_PER_STEP,
                sender: AMLSIM_BASE + rec[is].parse::<i64>().ok()?,
                receiver: AMLSIM_BASE + rec[ir].parse::<i64>().ok()?,
                amount,
                amount_gbp: amount * USD_GBP,
                pcur: "USD",
                rcur: "USD",
                scountry: country("US"),
                rcountry: country("US"),
                ptype: PaymentType::Ach,
                label: Some(flag),
                typology: flag.then(|| types.get(&alert).cloned().unwrap_or_else(|| "fraud".into())),
            })
        })());
    }
    Ok(rows)
}

// ------------------------------------------------------------------------------ FraudShield
fn fraudshield(path: &Path, limit: Option<usize>) -> Result<Vec<Option<Row>>> {
    let mut rdr = reader(path)?;
    let h = rdr.headers()?.clone();
    let (ic, ia, itm, id, ity, im, il, ih, iint, ifr) = (
        col(&h, "Customer_ID")?, col(&h, "Transaction_Amount (in Million)")?, col(&h, "Transaction_Time")?, col(&h, "Transaction_Date")?,
        col(&h, "Transaction_Type")?, col(&h, "Merchant_ID")?, col(&h, "Transaction_Location")?, col(&h, "Customer_Home_Location")?,
        col(&h, "Is_International_Transaction")?, col(&h, "Fraud_Label")?,
    );
    let mut rows = vec![];
    for rec in rdr.records().take(limit.unwrap_or(usize::MAX)) {
        let rec = rec?;
        rows.push((|| -> Option<Row> {
            let amount = rec[ia].parse::<f64>().ok()? * 1_000_000.0;
            let flag = &rec[ifr] == "Fraud";
            Some(Row {
                ts: parse_datetime(&rec[id], Some(&rec[itm]))?,
                sender: 400_000_000 + rec[ic].parse::<f64>().ok()? as i64,
                receiver: 3_000_000_000 + rec[im].parse::<f64>().ok()? as i64,
                amount,
                amount_gbp: amount * PKR_GBP,
                pcur: "PKR",
                rcur: if &rec[iint] == "Yes" { "USD" } else { "PKR" },
                scountry: country(city_country(&rec[ih])?),
                rcountry: country(city_country(&rec[il])?),
                ptype: PaymentType::parse(&rec[ity])?,
                label: Some(flag),
                typology: flag.then(|| "Fraud".to_string()),
            })
        })());
    }
    Ok(rows)
}

fn sibling(path: &Path, name: &str) -> Result<std::path::PathBuf> {
    let p = path.parent().unwrap_or(Path::new(".")).join(name);
    if p.exists() { Ok(p) } else { bail!("expected companion file {} next to {}", p.display(), path.display()) }
}

// ------------------------------------------------------------------ Synthetic KYC + transactions
fn kyc_risk(path: &Path, limit: Option<usize>) -> Result<(Vec<CustomerRow>, Vec<Option<Row>>)> {
    // clients: KYC attributes
    let mut crd = reader(&sibling(path, "clients_with_fatf_ofac.csv")?)?;
    let h = crd.headers()?.clone();
    let (cid, cty, csec, csr, cco, cpep, csan) = (
        col(&h, "client_id")?, col(&h, "client_type")?, col(&h, "sector")?, col(&h, "sector_risk")?,
        col(&h, "country")?, col(&h, "pep_flag")?, col(&h, "sanctions_flag")?,
    );
    let mut customers = vec![];
    for rec in crd.records() {
        let rec = rec?;
        let Some(id) = digits(&rec[cid]).map(|d| 200_000_000 + d) else { continue };
        let _ = cco;
        let mut c = CustomerRow::basic(
            id,
            format!("CLIENT-{}", id - 200_000_000),
            // industry risk raises the rating one notch; PEP/sanctions are handled by their own flags
            if &rec[csr] == "High" { CustomerRisk::Medium } else { CustomerRisk::Low },
            None,
        );
        c.customer_type = match &rec[cty] {
            "Corporate" => "corporate",
            "NGO" => "ngo",
            "Financial Institution" => "financial_institution",
            "Individual" => "individual",
            _ => "other",
        };
        c.sector = Some(rec[csec].to_string());
        c.pep = &rec[cpep] == "1";
        c.sanctioned = &rec[csan] == "1";
        customers.push(c);
    }
    // transactions
    let mut rdr = reader(path)?;
    let h = rdr.headers()?.clone();
    let (ia, ity, its, icl, icc, icp, io, ifa, ist, irp, itr) = (
        col(&h, "amount")?, col(&h, "transaction_type")?, col(&h, "timestamp")?, col(&h, "client_id")?, col(&h, "client_country")?,
        col(&h, "counterparty_country")?, col(&h, "ofac_match_flag")?, col(&h, "fatf_country_flag")?,
        col(&h, "structuring_pattern_flag")?, col(&h, "rapid_movement_flag")?, col(&h, "trade_mispricing_flag")?,
    );
    let iso = |c: &str| Country::parse(if c.trim() == "UK" { "GB" } else { c });
    let mut rows = vec![];
    for rec in rdr.records().take(limit.unwrap_or(usize::MAX)) {
        let rec = rec?;
        rows.push((|| -> Option<Row> {
            let amount: f64 = rec[ia].parse().ok()?;
            let cp = iso(&rec[icp])?;
            let typology = [(io, "OFAC_Match"), (ifa, "FATF_Country"), (ist, "Structuring_Pattern"), (itr, "Trade_Mispricing"), (irp, "Rapid_Movement")]
                .iter()
                .find(|(i, _)| &rec[*i] == "1")
                .map(|(_, n)| n.to_string());
            Some(Row {
                ts: parse_datetime(&rec[its], None)?,
                sender: 200_000_000 + digits(&rec[icl])?,
                receiver: 9_100_000_000 + ordinal(cp),
                amount,
                // amounts are in the reporting currency; no FX is applied
                amount_gbp: amount,
                pcur: "GBP",
                rcur: "GBP",
                scountry: iso(&rec[icc])?,
                rcountry: cp,
                ptype: PaymentType::parse(&rec[ity])?,
                label: Some(typology.is_some()),
                typology,
            })
        })());
    }
    Ok((customers, rows))
}

// ---------------------------------------------------------------------------- Lending
fn lending(path: &Path, limit: Option<usize>) -> Result<(Vec<CustomerRow>, Vec<Option<Row>>)> {
    let mut crd = reader(&sibling(path, "customer_financial_data.csv")?)?;
    let h = crd.headers()?.clone();
    let (cid, cinc) = (col(&h, "Customer_ID")?, col(&h, "Annual_Income")?);
    let mut customers = vec![];
    for rec in crd.records() {
        let rec = rec?;
        if let (Some(id), Ok(inc)) = (digits(&rec[cid]).map(|d| 300_000_000 + d), rec[cinc].parse::<f64>()) {
            customers.push(CustomerRow::basic(id, rec[cid].to_string(), CustomerRisk::Low, Some(inc)));
        }
    }
    let mut drd = reader(&sibling(path, "historical_default_data.csv")?)?;
    let h = drd.headers()?.clone();
    let (dl, dd) = (col(&h, "Loan_ID")?, col(&h, "Default")?);
    let mut defaults = HashMap::new();
    for rec in drd.records() {
        let rec = rec?;
        defaults.insert(rec[dl].to_string(), &rec[dd] == "1");
    }
    let mut rdr = reader(path)?;
    let h = rdr.headers()?.clone();
    let (il, ic, ia, id) = (col(&h, "Loan_ID")?, col(&h, "Customer_ID")?, col(&h, "Loan_Amount")?, col(&h, "Loan_Start_Date")?);
    let mut rows = vec![];
    for rec in rdr.records().take(limit.unwrap_or(usize::MAX)) {
        let rec = rec?;
        rows.push((|| -> Option<Row> {
            let amount: f64 = rec[ia].parse().ok()?;
            let bad = *defaults.get(&rec[il])?;
            Some(Row {
                ts: parse_datetime(&rec[id], None)?,
                sender: 4_000_000_000, // the lender's disbursement account
                receiver: 300_000_000 + digits(&rec[ic])?,
                amount,
                amount_gbp: amount,
                pcur: "GBP",
                rcur: "GBP",
                scountry: country("GB"),
                rcountry: country("GB"),
                ptype: PaymentType::OnlineTransfer,
                label: Some(bad),
                typology: bad.then(|| "Loan_Default".to_string()),
            })
        })());
    }
    Ok((customers, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("aml_ds_{}_{}", std::process::id(), name));
        std::fs::File::create(&p).unwrap().write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn amlsim_maps_steps_to_days_labels_and_typology() {
        let dir = std::env::temp_dir().join(format!("aml_ds_amlsim_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("transactions.csv"), "TX_ID,SENDER_ACCOUNT_ID,RECEIVER_ACCOUNT_ID,TX_TYPE,TX_AMOUNT,TIMESTAMP,IS_FRAUD,ALERT_ID\r\n2,5,6,TRANSFER,9.5,3,True,7\r\n1,1,2,TRANSFER,100.0,1,False,-1\r\n").unwrap();
        std::fs::write(dir.join("alerts.csv"), "ALERT_ID,ALERT_TYPE,IS_FRAUD,TX_ID,SENDER_ACCOUNT_ID,RECEIVER_ACCOUNT_ID,TX_TYPE,TX_AMOUNT,TIMESTAMP\r\n7,cycle,True,2,5,6,TRANSFER,9.5,3\r\n").unwrap();
        let v: Vec<Item> = open(&dir.join("transactions.csv"), Format::AmlSim, None).unwrap().collect();
        let rows: Vec<&Row> = v.iter().filter_map(|i| if let Item::Row(r) = i { Some(r) } else { None }).collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].ts, 86_400);
        assert_eq!(rows[0].label, Some(false));
        assert_eq!(rows[1].ts, 3 * 86_400);
        assert_eq!(rows[1].typology.as_deref(), Some("cycle"));
        assert_eq!(rows[1].sender, AMLSIM_BASE + 5);
    }

    #[test]
    fn datetime_variants() {
        assert_eq!(parse_datetime("2024-08-27 12:10:36.928669", None), parse_datetime("2024-08-27", Some("12:10:36")));
        assert_eq!(parse_datetime("2025-01-02T09:52:38Z", None), parse_datetime("2025-01-02 09:52:38", None));
        assert!(parse_datetime("garbage", None).is_none());
    }

    #[test]
    fn aml_suspicious_maps_customers_sorts_and_flags_bad_rows() {
        let p = tmp("aml.csv", "Transaction_ID,Customer_ID,Transaction_Date,Transaction_Type,Country,Amount,Customer_Risk_Profile,Annual_Income,Risk_Score,Suspicious_Flag\n\
            T1,CUST000002,2025-02-01 10:00:00.5,Wire Transfer,Iran,500.5,High,100000,90,1\n\
            T2,CUST000002,2025-01-01 10:00:00,Cash Deposit,Germany,10,High,100000,10,0\n\
            T3,CUST000009,2025-01-15 10:00:00,Cash Deposit,Atlantis,10,Low,5,10,0\n");
        let items: Vec<Item> = open(&p, Format::AmlSuspicious, None).unwrap().collect();
        let customers = items.iter().filter(|i| matches!(i, Item::Customer(_))).count();
        let bad = items.iter().filter(|i| matches!(i, Item::Bad)).count();
        let rows: Vec<&Row> = items.iter().filter_map(|i| if let Item::Row(r) = i { Some(r) } else { None }).collect();
        assert_eq!((customers, bad, rows.len()), (1, 1, 2));
        assert!(rows[0].ts < rows[1].ts, "rows must be chronological");
        assert_eq!(rows[1].rcountry.as_str(), "IR");
        assert_eq!(rows[1].label, Some(true));
        assert_eq!(rows[0].sender, 100_000_002);
        match &items[0] {
            Item::Customer(c) => assert_eq!((c.risk, c.account), (CustomerRisk::High, 100_000_002)),
            _ => panic!("customers come first"),
        }
    }

    #[test]
    fn saml_rows_stream_lazily_and_respect_limit() {
        let p = tmp("saml.csv", "Time,Date,Sender_account,Receiver_account,Amount,Payment_currency,Received_currency,Sender_bank_location,Receiver_bank_location,Payment_type,Is_laundering,Laundering_type\n\
            10:35:19,2022-10-07,1,2,1459.15,UK pounds,UK pounds,UK,UK,Cash Deposit,0,Normal_Cash_Deposits\n\
            10:35:20,2022-10-07,3,4,6019.64,UK pounds,Dirham,UK,UAE,Cross-border,1,Structuring\n\
            10:35:21,2022-10-07,5,6,1.0,UK pounds,UK pounds,UK,UK,ACH,0,Normal\n");
        let v: Vec<Item> = open(&p, Format::SamlD, Some(2)).unwrap().collect();
        assert_eq!(v.len(), 2);
        match &v[1] {
            Item::Row(r) => assert_eq!((r.label, r.typology.as_deref(), r.rcountry.as_str()), (Some(true), Some("Structuring"), "AE")),
            _ => panic!(),
        }
    }

    #[test]
    fn fraudshield_and_risk_prediction_map_accounts_into_disjoint_namespaces() {
        let p = tmp("fs.csv", "Transaction_ID,Customer_ID,Transaction_Amount (in Million),Transaction_Time,Transaction_Date,Transaction_Type,Merchant_ID,Merchant_Category,Transaction_Location,Customer_Home_Location,Is_International_Transaction,Fraud_Label\n\
            1.0,24239.0,6.0,10:54,2025-03-08,POS,97028.0,ATM,Singapore,Lahore,Yes,Fraud\n");
        match open(&p, Format::FraudShield, None).unwrap().next().unwrap() {
            Item::Row(r) => {
                assert_eq!((r.sender, r.receiver, r.rcountry.as_str(), r.rcur), (400_024_239, 3_000_097_028, "SG", "USD"));
                assert!((r.amount_gbp - 16_800.0).abs() < 1.0);
            }
            _ => panic!(),
        }
        let p = tmp("rp.csv", "Transaction ID,Timestamp,Sender Account ID,Receiver Account ID,Amount,Type,Device ID,IP Address,Location,Account Balance,Suspicious Activity Flag\n\
            T1,2025-01-02 09:52:38,A41624,B51095,100.0,Transfer,D1,1.1.1.1,Florida,5,1\n");
        match open(&p, Format::RiskPrediction, None).unwrap().next().unwrap() {
            Item::Row(r) => assert_eq!((r.sender, r.receiver, r.ptype), (1_000_041_624, 2_000_051_095, PaymentType::OnlineTransfer)),
            _ => panic!(),
        }
    }

    #[test]
    fn kyc_risk_joins_clients_and_prioritises_flag_typologies() {
        let dir = std::env::temp_dir().join(format!("aml_kyc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("clients_with_fatf_ofac.csv"), "client_id,client_name,client_type,sector,sector_risk,country,pep_flag,sanctions_flag,fatf_country_flag,ofac_country_flag,sectoral_sanctions_flag,ownership_opacity_score\n\
            7,Acme,Corporate,Import/Export,High,CH,1,1,0,0,0,0.0\n8,Bob,Individual,Retail,Low,FR,0,0,0,0,0,0.0\n").unwrap();
        let txf = dir.join("transactions_with_fatf_ofac.csv");
        std::fs::write(&txf, "transaction_id,client_id,amount,transaction_type,timestamp,client_country,counterparty_country,ofac_match_flag,fatf_country_flag,structuring_pattern_flag,rapid_movement_flag,trade_mispricing_flag\n\
            1,7,9500.5,SWIFT,2025-08-20 09:21:56,CH,IR,1,1,0,0,0\n2,8,10.0,Check,2025-08-19 09:00:00,FR,UK,0,0,0,0,0\n").unwrap();
        let items: Vec<Item> = open(&txf, Format::KycRisk, None).unwrap().collect();
        let c7 = items.iter().find_map(|i| match i { Item::Customer(c) if c.account == 200_000_007 => Some(c), _ => None }).unwrap();
        assert!(c7.pep && c7.sanctioned && c7.sector.as_deref() == Some("Import/Export") && c7.customer_type == "corporate");
        assert_eq!(c7.risk, CustomerRisk::Medium);
        let rows: Vec<&Row> = items.iter().filter_map(|i| if let Item::Row(r) = i { Some(r) } else { None }).collect();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].ts < rows[1].ts && rows[0].label == Some(false));
        assert_eq!(rows[1].typology.as_deref(), Some("OFAC_Match"), "OFAC outranks FATF when both are set");
        assert_eq!((rows[1].ptype, rows[0].rcountry.as_str()), (PaymentType::CrossBorder, "GB"));
    }

    #[test]
    fn lending_maps_disbursements_with_income_and_default_labels() {
        let dir = std::env::temp_dir().join(format!("aml_lend_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("customer_financial_data.csv"), "Customer_ID,Name,Age,Gender,Annual_Income,Credit_Score,Total_Debt,Employment_Status\nC00000005,X,40,Male,50000.0,700,1.0,Employed\n").unwrap();
        std::fs::write(dir.join("historical_default_data.csv"), "Loan_ID,Default\nL1,1\n").unwrap();
        let lf = dir.join("loan_information_data.csv");
        std::fs::write(&lf, "Loan_ID,Customer_ID,Loan_Amount,Interest_Rate,Loan_Term,Loan_Start_Date\nL1,C00000005,40000.0,5.0,60 months,2024-06-04\nL2,C00000005,1.0,5.0,60 months,2024-06-05\n").unwrap();
        let items: Vec<Item> = open(&lf, Format::Lending, None).unwrap().collect();
        match &items[0] {
            Item::Customer(c) => assert_eq!((c.account, c.annual_income_gbp), (300_000_005, Some(50000.0))),
            _ => panic!(),
        }
        assert_eq!(items.iter().filter(|i| matches!(i, Item::Bad)).count(), 1, "loan without a default record is unmappable, not silently kept");
        let r = items.iter().find_map(|i| if let Item::Row(r) = i { Some(r) } else { None }).unwrap();
        assert_eq!((r.receiver, r.label), (300_000_005, Some(true)));
    }
}
