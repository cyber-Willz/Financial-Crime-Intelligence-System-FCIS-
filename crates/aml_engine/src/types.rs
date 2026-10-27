use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaymentType {
    CashDeposit,
    CashWithdrawal,
    CrossBorder,
    Ach,
    Cheque,
    CreditCard,
    DebitCard,
    Wire,
    OnlineTransfer,
    Pos,
    Atm,
}

impl PaymentType {
    /// Lenient parser: "Cash Deposit", "CASH_DEPOSIT", "cash-deposit" all work.
    pub fn parse(s: &str) -> Option<Self> {
        let k: String = s
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        Some(match k.as_str() {
            "cashdeposit" => Self::CashDeposit,
            "cashwithdrawal" => Self::CashWithdrawal,
            "crossborder" => Self::CrossBorder,
            "ach" => Self::Ach,
            "cheque" | "check" | "chequedeposit" => Self::Cheque,
            "foreignremittance" => Self::CrossBorder,
            "wire" | "wiretransfer" => Self::Wire,
            "swift" => Self::CrossBorder,
            "online" | "onlinetransfer" => Self::OnlineTransfer,
            "pos" => Self::Pos,
            "atm" => Self::Atm,
            "creditcard" => Self::CreditCard,
            "debitcard" => Self::DebitCard,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CashDeposit => "CASH_DEPOSIT",
            Self::CashWithdrawal => "CASH_WITHDRAWAL",
            Self::CrossBorder => "CROSS_BORDER",
            Self::Ach => "ACH",
            Self::Cheque => "CHEQUE",
            Self::CreditCard => "CREDIT_CARD",
            Self::DebitCard => "DEBIT_CARD",
            Self::Wire => "WIRE",
            Self::OnlineTransfer => "ONLINE_TRANSFER",
            Self::Pos => "POS",
            Self::Atm => "ATM",
        }
    }
    pub fn is_cash(&self) -> bool {
        matches!(self, Self::CashDeposit | Self::CashWithdrawal)
    }
}

/// ISO-3166 alpha-2 country code, stored as two upper-case ASCII bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Country(pub [u8; 2]);

impl Country {
    pub fn parse(s: &str) -> Option<Self> {
        let b = s.trim().as_bytes();
        if b.len() == 2 && b.iter().all(|c| c.is_ascii_alphabetic()) {
            Some(Country([b[0].to_ascii_uppercase(), b[1].to_ascii_uppercase()]))
        } else {
            None
        }
    }
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).unwrap_or("??")
    }
    pub(crate) fn index(&self) -> usize {
        let a = self.0[0].wrapping_sub(b'A') as usize;
        let b = self.0[1].wrapping_sub(b'A') as usize;
        if a < 26 && b < 26 {
            a * 26 + b
        } else {
            0
        }
    }
}

impl Serialize for Country {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}
impl<'de> Deserialize<'de> for Country {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Country::parse(&s).ok_or_else(|| serde::de::Error::custom("invalid ISO-3166 alpha-2 code"))
    }
}

/// A transaction as seen by the engine. Money is already normalised to GBP.
#[derive(Debug, Clone)]
pub struct TxnIn {
    pub ts: i64,
    pub sender: i64,
    pub receiver: i64,
    pub amount_gbp: f64,
    pub payment_type: PaymentType,
    pub sender_country: Country,
    pub receiver_country: Country,
    pub cross_currency: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Sender,
    Receiver,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub rule: &'static str,
    pub weight: f64,
    pub side: Side,
    pub detail: String,
}

/// A *previous* transaction that this evaluation retroactively links to a pattern
/// (e.g. the earlier legs of a fan-in that just crossed its threshold).
#[derive(Debug, Clone, Serialize)]
pub struct Linked {
    pub rf: u32,
    pub rule: &'static str,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Evaluation {
    pub score: u8,
    pub hits: Vec<Hit>,
    pub linked: Vec<Linked>,
}

impl Evaluation {
    pub fn rule_codes(&self) -> Vec<&'static str> {
        self.hits.iter().map(|h| h.rule).collect()
    }
    /// The account an analyst should look at first: the side of the heaviest hit.
    pub fn subject_side(&self) -> Side {
        self.hits
            .iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
            .map(|h| h.side)
            .unwrap_or(Side::Sender)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl Severity {
    pub fn from_score(score: u8) -> Self {
        match score {
            90..=u8::MAX => Severity::Critical,
            75..=89 => Severity::High,
            60..=74 => Severity::Medium,
            _ => Severity::Low,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Low => "low",
            Severity::Medium => "medium",
            Severity::High => "high",
            Severity::Critical => "critical",
        }
    }
}

/// KYC / customer due-diligence rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CustomerRisk {
    Low,
    Medium,
    High,
}

impl CustomerRisk {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// What the engine needs to know about the customer behind an account.
#[derive(Debug, Clone, Copy)]
pub struct Profile {
    pub risk: CustomerRisk,
    /// Declared annual income in GBP, if known.
    pub annual_income_gbp: Option<f64>,
    /// Politically exposed persons are treated as high risk regardless of rating.
    pub pep: bool,
    /// The customer's sector is trade-based (import/export): enables trade-mispricing monitoring.
    pub trade_exposed: bool,
}

/// Import/export (trade-based) sectors are monitored for trade mispricing.
pub fn is_trade_sector(sector: &str) -> bool {
    let s = sector.to_ascii_lowercase();
    s.contains("import") || s.contains("export")
}
