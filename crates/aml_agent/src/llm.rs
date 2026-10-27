//! The language-model seam. The agent only needs "prompt in, text out"; where the text comes from is replaceable.
//!
//! * [`ScriptedLlm`] — canned replies for tests.
//! * [`ProcessLlm`] — a child process speaking JSON lines on stdin/stdout (the `llm_sidecar` crate wraps
//!   `burn-llm` this way). Inference runs out-of-process so a slow or crashing model cannot take the
//!   transaction pipeline with it; each request has a timeout, after which the child is killed and restarted
//!   on the next call. Protocol (one JSON object per line):
//!   `→ {"id":1,"system":"…","user":"…","max_new":256}`   `← {"id":1,"ok":true,"text":"…"}` or `{"id":1,"ok":false,"error":"…"}`;
//!   the child first prints `{"ready":true,"model":"name","max_context":4096}`.

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

pub trait Llm: Send {
    fn name(&self) -> String;
    fn complete(&mut self, system: &str, user: &str, max_new: usize) -> Result<String, String>;
}

/// Replies from a queue; records every prompt it was given.
pub struct ScriptedLlm {
    pub replies: VecDeque<String>,
    pub prompts: Vec<(String, String)>,
}

impl ScriptedLlm {
    pub fn new(replies: &[&str]) -> Self {
        ScriptedLlm { replies: replies.iter().map(|s| s.to_string()).collect(), prompts: vec![] }
    }
}

impl Llm for ScriptedLlm {
    fn name(&self) -> String {
        "scripted".into()
    }
    fn complete(&mut self, system: &str, user: &str, _max_new: usize) -> Result<String, String> {
        self.prompts.push((system.to_string(), user.to_string()));
        self.replies.pop_front().ok_or_else(|| "script exhausted".to_string())
    }
}

struct Live {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<String>,
    model: String,
}

pub struct ProcessLlm {
    cmd: String,
    args: Vec<String>,
    pub request_timeout: Duration,
    pub startup_timeout: Duration,
    live: Option<Live>,
    next_id: u64,
    label: String,
}

impl ProcessLlm {
    pub fn new(cmd: impl Into<String>, args: Vec<String>) -> Self {
        let cmd = cmd.into();
        ProcessLlm { label: cmd.clone(), cmd, args, request_timeout: Duration::from_secs(120), startup_timeout: Duration::from_secs(180), live: None, next_id: 0 }
    }

    fn start(&mut self) -> Result<(), String> {
        let mut child = Command::new(&self.cmd)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("cannot start LLM process '{}': {e}", self.cmd))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let out = child.stdout.take().ok_or("no stdout")?;
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for l in BufReader::new(out).lines() {
                match l {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let deadline = Instant::now() + self.startup_timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(left) {
                Ok(l) => {
                    if let Ok(v) = serde_json::from_str::<Value>(&l) {
                        if v.get("ready").and_then(Value::as_bool) == Some(true) {
                            let model = v.get("model").and_then(Value::as_str).unwrap_or("unknown").to_string();
                            self.label = model.clone();
                            self.live = Some(Live { child, stdin, rx, model });
                            return Ok(());
                        }
                    }
                }
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("LLM process did not report ready (model failed to load or timed out)".into());
                }
            }
        }
    }

    fn drop_live(&mut self) {
        if let Some(mut l) = self.live.take() {
            let _ = l.child.kill();
            let _ = l.child.wait();
        }
    }
}

impl Drop for ProcessLlm {
    fn drop(&mut self) {
        self.drop_live();
    }
}

impl Llm for ProcessLlm {
    fn name(&self) -> String {
        self.live.as_ref().map(|l| l.model.clone()).unwrap_or_else(|| self.label.clone())
    }

    fn complete(&mut self, system: &str, user: &str, max_new: usize) -> Result<String, String> {
        if self.live.is_none() {
            self.start()?;
        }
        self.next_id += 1;
        let id = self.next_id;
        let req = json!({"id": id, "system": system, "user": user, "max_new": max_new}).to_string();
        let timeout = self.request_timeout;
        let res: Result<String, String> = {
            let live = self.live.as_mut().unwrap();
            if writeln!(live.stdin, "{req}").and_then(|_| live.stdin.flush()).is_err() {
                Err("LLM process closed its input".to_string())
            } else {
                let deadline = Instant::now() + timeout;
                loop {
                    let left = deadline.saturating_duration_since(Instant::now());
                    match live.rx.recv_timeout(left) {
                        Ok(l) => match serde_json::from_str::<Value>(&l) {
                            Ok(v) if v.get("id").and_then(Value::as_u64) == Some(id) => {
                                break if v.get("ok").and_then(Value::as_bool) == Some(true) {
                                    Ok(v.get("text").and_then(Value::as_str).unwrap_or("").to_string())
                                } else {
                                    Err(v.get("error").and_then(Value::as_str).unwrap_or("LLM error").to_string())
                                };
                            }
                            _ => continue, // stray output: ignore
                        },
                        Err(_) => break Err(format!("LLM did not answer within {} s", timeout.as_secs())),
                    }
                }
            }
        };
        // a failure that may leave the child in an unknown state: restart it next time
        if matches!(&res, Err(e) if e.contains("did not answer") || e.contains("closed")) {
            self.drop_live();
        }
        res
    }
}
