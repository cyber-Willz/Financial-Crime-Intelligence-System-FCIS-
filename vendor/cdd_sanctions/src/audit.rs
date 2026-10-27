//! Append-only, hash-chained audit log (tamper-evident).

use crate::model::sha256_hex;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEvent {
    pub seq: u64,
    pub at: DateTime<Utc>,
    pub actor: String,
    pub action: String,
    pub target: String,
    pub detail: serde_json::Value,
    pub prev_hash: String,
    pub hash: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AuditLog {
    events: Vec<AuditEvent>,
}

fn digest(prev: &str, seq: u64, at: &DateTime<Utc>, actor: &str, action: &str, target: &str, detail: &serde_json::Value) -> String {
    let d = serde_json::to_string(detail).unwrap_or_default();
    sha256_hex(format!("{prev}|{seq}|{}|{actor}|{action}|{target}|{d}", at.to_rfc3339()).as_bytes())
}

impl AuditLog {
    pub fn append(&mut self, at: DateTime<Utc>, actor: &str, action: &str, target: &str, detail: serde_json::Value) {
        let seq = self.events.len() as u64 + 1;
        let prev_hash = self.events.last().map(|e| e.hash.clone()).unwrap_or_else(|| "genesis".to_string());
        let hash = digest(&prev_hash, seq, &at, actor, action, target, &detail);
        self.events.push(AuditEvent {
            seq,
            at,
            actor: actor.into(),
            action: action.into(),
            target: target.into(),
            detail,
            prev_hash,
            hash,
        });
    }

    pub fn events(&self) -> &[AuditEvent] {
        &self.events
    }

    /// `Err(seq)` identifies the first event whose content or chain link is wrong.
    pub fn verify(&self) -> std::result::Result<(), u64> {
        let mut prev = "genesis".to_string();
        for (i, e) in self.events.iter().enumerate() {
            if e.seq != i as u64 + 1 || e.prev_hash != prev {
                return Err(e.seq);
            }
            if digest(&e.prev_hash, e.seq, &e.at, &e.actor, &e.action, &e.target, &e.detail) != e.hash {
                return Err(e.seq);
            }
            prev = e.hash.clone();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn log3() -> AuditLog {
        let mut l = AuditLog::default();
        let t = Utc::now();
        l.append(t, "a", "x", "p1", json!({"k": 1}));
        l.append(t, "b", "y", "p2", json!({"k": 2}));
        l.append(t, "c", "z", "p3", json!({"k": 3}));
        l
    }

    #[test]
    fn verifies_and_detects_tampering() {
        let l = log3();
        assert_eq!(l.verify(), Ok(()));

        let mut edited = l.clone();
        edited.events[1].detail = json!({"k": 999});
        assert_eq!(edited.verify(), Err(2));

        let mut deleted = l.clone();
        deleted.events.remove(1);
        assert!(deleted.verify().is_err());

        let mut roundtrip: AuditLog = serde_json::from_str(&serde_json::to_string(&l).unwrap()).unwrap();
        assert_eq!(roundtrip.verify(), Ok(()));
        roundtrip.events[0].actor = "mallory".into();
        assert_eq!(roundtrip.verify(), Err(1));
    }
}
