//! `aml-llm-sidecar <model_dir> [--no-think] [--max-context N]`
//!
//! Loads a Llama/Qwen safetensors checkpoint with burn-llm (CPU, f32) and answers JSON-line requests:
//!   <- {"id":1,"system":"…","user":"…","max_new":256}
//!   -> {"id":1,"ok":true,"text":"…"}   |   {"id":1,"ok":false,"error":"…"}
//! Startup line: {"ready":true,"model":"<dir name>","max_context":N}. Decoding is greedy (temperature 0) so the same
//! prompt gives the same answer. stdout carries protocol lines only; diagnostics go to stderr.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use burn_ndarray::{NdArray, NdArrayDevice};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};

use burn_llm::chat::{self, ChatOptions, Message};
use burn_llm::config::ModelConfig;
use burn_llm::generate::{generate, SamplingConfig};
use burn_llm::tokenizer::{load_tokenizer, resolve_stop_ids, TokenStream};
use burn_llm::weights::load_model;

type B = NdArray<f32>;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir: PathBuf = args.next().context("usage: aml-llm-sidecar <model_dir> [--no-think] [--max-context N]")?.into();
    let (mut no_think, mut max_context) = (false, 4096usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--no-think" => no_think = true,
            "--max-context" => max_context = args.next().context("--max-context needs a value")?.parse()?,
            o => bail!("unknown option {o}"),
        }
    }
    let device = NdArrayDevice::default();
    let model_cfg = ModelConfig::from_json_str(&std::fs::read_to_string(dir.join("config.json")).context("config.json")?)?;
    let tok = load_tokenizer(&dir)?;
    let stop_ids = resolve_stop_ids(&dir, &model_cfg, &tok);
    let model = load_model::<B>(&dir, None, &device)?;
    let fmt = chat::detect_format(&tok);
    let name = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    eprintln!("loaded {name}: {} layers, chat template: {}", model.num_layers(), fmt.is_some());

    let out = std::io::stdout();
    let mut out = out.lock();
    writeln!(out, "{}", json!({"ready": true, "model": name, "max_context": max_context}))?;
    out.flush()?;

    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let Ok(req) = serde_json::from_str::<Value>(&line) else { continue };
        let id = req["id"].clone();
        let reply = (|| -> Result<String> {
            let system = req["system"].as_str().unwrap_or("");
            let user = req["user"].as_str().context("missing user")?;
            let max_new = req["max_new"].as_u64().unwrap_or(256).clamp(1, 1024) as usize;
            let (text, add_special) = match fmt {
                Some(f) => {
                    let mut msgs = vec![];
                    if !system.is_empty() {
                        msgs.push(Message::new("system", system));
                    }
                    msgs.push(Message::new("user", user));
                    (chat::render(f, &msgs, &ChatOptions { no_think }), false)
                }
                None => (format!("{system}\n\n{user}\n\nANSWER:"), true),
            };
            let ids = tok.encode(&text, add_special)?;
            if ids.len() + max_new > max_context {
                bail!("prompt of {} tokens plus {} new exceeds the context limit {}", ids.len(), max_new, max_context);
            }
            let cfg = SamplingConfig { max_new_tokens: max_new, temperature: 0.0, ..Default::default() };
            let mut rng = StdRng::seed_from_u64(0);
            let mut stream = TokenStream::new(&tok).with_context(&ids);
            let mut s = String::new();
            generate(&model, &ids, &stop_ids, &cfg, &device, &mut rng, |t| {
                if let Ok(Some(p)) = stream.push(t) {
                    s.push_str(&p);
                }
            })?;
            if let Ok(Some(rest)) = stream.flush() {
                s.push_str(&rest);
            }
            // a template-less base model continues after "ANSWER:"; make the reply parseable
            if fmt.is_none() && !s.contains("ANSWER:") {
                s = format!("ANSWER: {}", s.trim());
            }
            Ok(s)
        })();
        let msg = match reply {
            Ok(t) => json!({"id": id, "ok": true, "text": t}),
            Err(e) => json!({"id": id, "ok": false, "error": format!("{e:#}")}),
        };
        writeln!(out, "{msg}")?;
        out.flush()?;
    }
    Ok(())
}
