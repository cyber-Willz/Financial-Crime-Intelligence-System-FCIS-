//! Test double for the LLM sidecar protocol (see `llm.rs`). Mode from `FAKE_LLM_MODE`:
//! `good` (cited, grounded answer), `garbage` (invented number, bad citation, accusation),
//! `query` (first asks for a graph query, then answers), `hang` (never answers).
use serde_json::{json, Value};
use std::io::{BufRead, Write};

fn main() {
    let mode = std::env::var("FAKE_LLM_MODE").unwrap_or_else(|_| "good".into());
    let out = std::io::stdout();
    let mut o = out.lock();
    let _ = writeln!(o, "{}", json!({"ready": true, "model": format!("fake-{mode}"), "max_context": 4096}));
    let _ = o.flush();
    let mut n = 0;
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        n += 1;
        let id = v["id"].clone();
        let text = match mode.as_str() {
            "hang" => loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            },
            "garbage" => "ANSWER: The holder is guilty and moved 98765 GBP [F99].".to_string(),
            "query" if n == 1 => "QUERY counterparties account=1 dir=in top=3".to_string(),
            "query" => "ANSWER: The alert was raised on the account [F1].".to_string(),
            _ => "ANSWER: The alert was raised on the account [F1].".to_string(),
        };
        let _ = writeln!(o, "{}", json!({"id": id, "ok": true, "text": text}));
        let _ = o.flush();
    }
}
