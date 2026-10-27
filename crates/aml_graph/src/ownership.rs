//! Beneficial-ownership context for the graph model.
//!
//! This is plain data: `aml_bo` derives it from the customer-due-diligence engine (`cdd_sanctions`) and hands
//! it to the model, so `aml_graph` itself does not depend on the CDD crate. For every account it records who
//! stands behind the holder (beneficial owners, ownership layers, opacity, sanctions exposure) and which
//! *ownership group* the holder belongs to (parties connected by ownership or control).

use std::collections::HashMap;

/// What is known about the party behind one account.
#[derive(Clone, Debug, Default)]
pub struct OwnerInfo {
    /// holder party id in the CDD system
    pub party: u64,
    /// ownership/control group (0 = none). Accounts of parties in one group share beneficial owners or control.
    pub group: u32,
    /// sorted ids of natural persons with a material stake or control (includes the holder when it is a person)
    pub owners: Vec<u64>,
    pub layers: u8,
    /// share of the ownership that stops at a corporate owner with no recorded owners (0..1)
    pub unresolved: f32,
    /// 0 (transparent) .. 1 (opaque): layers, unresolved owners, nominees, secrecy jurisdictions, cycles, fragmentation
    pub opacity: f32,
    /// share of the holder owned (directly or not) by blocked parties (0..1): the 50% rule input
    pub sanctioned_pct: f32,
    /// indirect sanctions exposure from the spectral network analysis (random-walk mass share, small numbers)
    pub exposure: f32,
    pub pep: bool,
    pub high_risk_jurisdiction: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    None,
    /// two accounts of the same holder
    SameParty,
    /// different holders with at least one beneficial owner in common (or one owns the other)
    SharedOwner,
    /// different holders connected through ownership/control, without a common material owner
    SameGroup,
}

#[derive(Clone, Debug, Default)]
pub struct OwnershipMap {
    accounts: HashMap<i64, OwnerInfo>,
}

impl OwnershipMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, account: i64, info: OwnerInfo) {
        self.accounts.insert(account, info);
    }
    pub fn get(&self, account: i64) -> Option<&OwnerInfo> {
        self.accounts.get(&account)
    }
    pub fn len(&self) -> usize {
        self.accounts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.accounts.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&i64, &OwnerInfo)> {
        self.accounts.iter()
    }

    pub fn relation(&self, a: i64, b: i64) -> Relation {
        if a == b {
            return Relation::None;
        }
        let (Some(x), Some(y)) = (self.accounts.get(&a), self.accounts.get(&b)) else { return Relation::None };
        if x.party == y.party {
            return Relation::SameParty;
        }
        // sorted-list intersection
        let (mut i, mut j) = (0, 0);
        while i < x.owners.len() && j < y.owners.len() {
            match x.owners[i].cmp(&y.owners[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => return Relation::SharedOwner,
            }
        }
        if x.group != 0 && x.group == y.group {
            Relation::SameGroup
        } else {
            Relation::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(party: u64, group: u32, owners: &[u64]) -> OwnerInfo {
        OwnerInfo { party, group, owners: owners.to_vec(), ..Default::default() }
    }

    #[test]
    fn relations() {
        let mut m = OwnershipMap::new();
        m.insert(1, info(10, 1, &[100]));
        m.insert(2, info(10, 1, &[100]));
        m.insert(3, info(11, 1, &[100, 101]));
        m.insert(4, info(12, 1, &[102]));
        m.insert(5, info(13, 2, &[103]));
        assert_eq!(m.relation(1, 2), Relation::SameParty);
        assert_eq!(m.relation(1, 3), Relation::SharedOwner);
        assert_eq!(m.relation(1, 4), Relation::SameGroup);
        assert_eq!(m.relation(1, 5), Relation::None);
        assert_eq!(m.relation(1, 99), Relation::None);
        assert_eq!(m.relation(1, 1), Relation::None);
    }
}
