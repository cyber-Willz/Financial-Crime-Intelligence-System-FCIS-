//! Which party holds which account.

use anyhow::{bail, Context, Result};
use cdd_sanctions::cdd::PartyId;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Account -> holder. One holder per account (joint accounts are out of scope: record the primary holder or
/// the legal entity the account belongs to).
#[derive(Clone, Debug, Default)]
pub struct AccountRegistry {
    by_account: HashMap<i64, PartyId>,
}

impl AccountRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, account: i64, party: PartyId) {
        self.by_account.insert(account, party);
    }
    pub fn party_of(&self, account: i64) -> Option<PartyId> {
        self.by_account.get(&account).copied()
    }
    pub fn len(&self) -> usize {
        self.by_account.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_account.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (i64, PartyId)> + '_ {
        self.by_account.iter().map(|(a, p)| (*a, *p))
    }
    pub fn accounts_by_party(&self) -> BTreeMap<PartyId, Vec<i64>> {
        let mut m: BTreeMap<PartyId, Vec<i64>> = BTreeMap::new();
        for (a, p) in self.iter() {
            m.entry(p).or_default().push(a);
        }
        for v in m.values_mut() {
            v.sort_unstable();
        }
        m
    }

    /// CSV with a header containing `account` and `party` columns.
    pub fn from_csv(path: &Path) -> Result<Self> {
        let mut rdr = csv::ReaderBuilder::new().trim(csv::Trim::All).from_path(path).with_context(|| format!("opening {}", path.display()))?;
        let h = rdr.headers()?.clone();
        let find = |n: &str| h.iter().position(|x| x.eq_ignore_ascii_case(n));
        let (Some(ia), Some(ip)) = (find("account"), find("party")) else { bail!("{}: need `account` and `party` columns", path.display()) };
        let mut r = Self::new();
        for (n, rec) in rdr.records().enumerate() {
            let rec = rec?;
            let a: i64 = rec[ia].parse().with_context(|| format!("row {}: bad account", n + 2))?;
            let p: PartyId = rec[ip].parse().with_context(|| format!("row {}: bad party", n + 2))?;
            if let Some(old) = r.by_account.insert(a, p) {
                if old != p {
                    bail!("row {}: account {a} is assigned to two parties ({old} and {p})", n + 2);
                }
            }
        }
        Ok(r)
    }

    pub fn write_csv(&self, path: &Path) -> Result<()> {
        let mut w = csv::Writer::from_path(path)?;
        w.write_record(["account", "party"])?;
        let mut v: Vec<(i64, PartyId)> = self.iter().collect();
        v.sort_unstable();
        for (a, p) in v {
            w.write_record([a.to_string(), p.to_string()])?;
        }
        w.flush()?;
        Ok(())
    }
}
