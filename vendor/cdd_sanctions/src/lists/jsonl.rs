//! Canonical JSON-lines import: one [`ListEntry`] per line. Use this for any
//! government or territory list that has no dedicated parser — convert it to
//! this shape (names, dobs, nationalities, identifiers, programs) and load it.

use crate::error::{Error, Result};
use crate::model::ListEntry;

pub fn parse(text: &str, list_name: &str) -> Result<Vec<ListEntry>> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut e: ListEntry = serde_json::from_str(line)
            .map_err(|x| Error::Invalid(format!("line {}: {x}", n + 1)))?;
        if e.names.is_empty() {
            return Err(Error::Invalid(format!("line {}: entry has no names", n + 1)));
        }
        e.list = list_name.to_string();
        out.push(e.finalize());
    }
    Ok(out)
}
