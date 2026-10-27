//! The agent: grounded explanations, constrained investigation, SAR drafting and queue summaries.

use crate::evidence::{Evidence, Fact, FactBook, FactIndex, NoteFact};
use crate::grounding::{check, GroundingReport};
use crate::llm::Llm;
use crate::query::{parse_query, run_query, GraphSource, Query};
use crate::typology::{self, identify, TypologyHit};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const DISCLAIMER: &str = "Advisory output for a human analyst. These are indicators that warrant review, not findings of wrongdoing. Nothing here has changed an alert, case, rule or threshold.";

const SYSTEM: &str = "You are an anti-money-laundering analyst assistant. Use ONLY the numbered FACTS. \
To look something up, reply with one or more lines, each exactly: QUERY <name> key=value ... \
Allowed names: neighbors, counterparties, flows, cycles, fan_in, fan_out, pass_through, ownership, history. Keys: account, a, b, depth, dir, top, days, max_len (integers). \
When you have enough, reply with one line starting ANSWER: followed by a short paragraph. Cite facts like [F3]. \
Use only numbers that appear in the facts. Describe indicators; never state that someone is guilty or committed a crime.";

const SYSTEM_TEXT: &str = "You are an anti-money-laundering analyst assistant. Use ONLY the numbered FACTS. \
Reply with one line starting ANSWER: followed by the requested text. Cite facts like [F3]. \
Use only numbers that appear in the facts. Describe indicators; never state that someone is guilty or committed a crime.";

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub max_steps: usize,
    pub max_queries: usize,
    pub max_new_tokens: usize,
    pub max_prompt_facts: usize,
    pub fact_chars: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig { max_steps: 3, max_queries: 6, max_new_tokens: 256, max_prompt_facts: 40, fact_chars: 260 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Provenance {
    /// model that produced the final text, when it did
    pub model: Option<String>,
    pub llm_used: bool,
    pub llm_steps: usize,
    pub grounding: GroundingReport,
    /// why the model's output was not used (unreachable, unparsable, failed grounding)
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutedQuery {
    pub query: String,
    pub ok: bool,
    pub note: Option<String>,
    pub fact_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Investigation {
    pub question: String,
    pub answer: String,
    pub facts: Vec<Fact>,
    pub queries: Vec<ExecutedQuery>,
    pub typologies: Vec<TypologyHit>,
    pub provenance: Provenance,
    pub disclaimer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Explanation {
    pub alert_id: Option<i64>,
    pub headline: String,
    pub narrative: String,
    pub typologies: Vec<TypologyHit>,
    pub key_facts: Vec<Fact>,
    pub next_steps: Vec<String>,
    pub queries: Vec<ExecutedQuery>,
    pub provenance: Provenance,
    pub disclaimer: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CaseFacts {
    pub case_id: i64,
    pub title: String,
    pub priority: String,
    pub status: String,
    pub notes: Vec<NoteFact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SarSection {
    pub title: String,
    pub body: String,
    pub fact_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SarDraft {
    pub case_id: i64,
    pub status: String,
    pub sections: Vec<SarSection>,
    pub open_items: Vec<String>,
    pub facts: Vec<Fact>,
    pub provenance: Provenance,
    pub disclaimer: String,
}

pub struct Agent {
    llm: Option<Box<dyn Llm>>,
    pub cfg: AgentConfig,
}

enum Reply {
    Queries(Vec<String>),
    Answer(String),
    Neither,
}

fn parse_reply(text: &str) -> Reply {
    let cleaned: String = text.lines().filter(|l| !l.trim_start().starts_with("```")).collect::<Vec<_>>().join("\n");
    let qs: Vec<String> = cleaned.lines().map(|l| l.trim().trim_start_matches(['-', '*', ' '])).filter(|l| l.starts_with("QUERY ")).map(String::from).collect();
    if !qs.is_empty() {
        return Reply::Queries(qs);
    }
    if let Some(i) = cleaned.find("ANSWER:") {
        let a = cleaned[i + 7..].trim().to_string();
        if !a.is_empty() {
            return Reply::Answer(a);
        }
    }
    Reply::Neither
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { format!("{}…", s.chars().take(n).collect::<String>()) }
}

fn render_facts(book: &FactBook, cfg: &AgentConfig) -> String {
    book.facts.iter().take(cfg.max_prompt_facts).map(|f| format!("[{}] {}", f.id, clip(&f.text, cfg.fact_chars))).collect::<Vec<_>>().join("\n")
}

/// Queries worth running for this evidence and question, most useful first (deduplicated, at most four).
pub fn plan(ev: &Evidence, question: &str) -> Vec<Query> {
    let a = ev.subject;
    let q = question.to_lowercase();
    let has = |k: &[&str]| k.iter().any(|w| q.contains(w));
    let mut out: Vec<Query> = vec![];
    if has(&["cycle", "circular", "round", "loop"]) {
        out.push(Query::Cycles { account: a, max_len: 4 });
    }
    if has(&["owner", "ubo", "beneficial", "shell"]) {
        out.push(Query::Ownership { account: a });
    }
    if has(&["history", "previous", "earlier", "before"]) {
        out.push(Query::History { account: a });
    }
    if has(&["who", "counterpart", "sender", "paid by", "source"]) {
        out.push(Query::Counterparties { account: a, dir: crate::query::Dir::In, top: 5 });
    }
    for r in ev.rule_codes() {
        match r {
            "FAN_IN" | "STRUCTURING_HUB" | "STRUCTURING_BAND" | "REPEATED_PAIR" => {
                out.push(Query::FanIn { account: a, days: 30 });
                out.push(Query::Counterparties { account: a, dir: crate::query::Dir::In, top: 5 });
            }
            "FAN_OUT" => {
                out.push(Query::FanOut { account: a, days: 30 });
                out.push(Query::Counterparties { account: a, dir: crate::query::Dir::Out, top: 5 });
            }
            "PASS_THROUGH" | "LAYERING_CHAIN" => {
                out.push(Query::PassThrough { account: a });
                out.push(Query::Counterparties { account: a, dir: crate::query::Dir::Out, top: 5 });
            }
            "CIRCULAR_FLOW" => out.push(Query::Cycles { account: a, max_len: 4 }),
            _ => {}
        }
    }
    if out.is_empty() {
        out.push(Query::Neighbors { account: a, depth: 1 });
    }
    let mut seen = BTreeSet::new();
    out.retain(|q| seen.insert(q.render()));
    out.truncate(4);
    out
}

fn cite_for(rule: &str, ix: &FactIndex) -> Option<String> {
    if let Some((_, id)) = ix.rules.iter().find(|(c, _)| c == rule) {
        return Some(id.clone());
    }
    match rule {
        "graph_model_risk" => ix.graph.first().cloned(),
        "opaque_owner" | "material_sanctioned_stake" | "network_exposure" | "fragmented_ownership" => ix.ownership.first().cloned(),
        "WATCHLIST_HIT" => ix.sanctions.first().cloned(),
        _ => None,
    }
}

fn typology_sentence(hits: &[TypologyHit], ix: &FactIndex) -> String {
    let top: Vec<&TypologyHit> = hits.iter().take(3).collect();
    if top.is_empty() {
        return "The evidence does not match a known typology.".into();
    }
    let parts: Vec<String> = top
        .iter()
        .map(|t| {
            let c: BTreeSet<String> = t.supporting_rules.iter().filter_map(|r| cite_for(r, ix)).collect();
            format!("{}{}", t.name, if c.is_empty() { String::new() } else { format!(" {}", c.iter().map(|x| format!("[{x}]")).collect::<String>()) })
        })
        .collect();
    format!("The evidence is most consistent with {}.", parts.join(", "))
}

fn cited(book: &FactBook, id: &str) -> String {
    format!("{} [{}]", book.get(id).map(|f| f.text.trim_end_matches('.')).unwrap_or(""), id)
}

impl Agent {
    pub fn new(llm: Option<Box<dyn Llm>>, cfg: AgentConfig) -> Self {
        Agent { llm, cfg }
    }
    pub fn has_llm(&self) -> bool {
        self.llm.is_some()
    }
    pub fn deterministic() -> Self {
        Agent::new(None, AgentConfig::default())
    }
    pub fn set_llm(&mut self, llm: Option<Box<dyn Llm>>) {
        self.llm = llm;
    }

    fn exec(&self, line_or_q: Result<Query, String>, raw: &str, g: &dyn GraphSource, book: &mut FactBook) -> ExecutedQuery {
        match line_or_q {
            Ok(q) => {
                let r = run_query(&q, g);
                let ids: Vec<String> = r.facts.into_iter().map(|t| book.add(t)).collect();
                ExecutedQuery { query: r.query, ok: true, note: if r.truncated { Some("search hit its row/step limit".into()) } else { None }, fact_ids: ids }
            }
            Err(e) => ExecutedQuery { query: clip(raw, 120), ok: false, note: Some(format!("refused: {e}")), fact_ids: vec![] },
        }
    }

    /// Ask the model for text built only from `facts`; returns the grounded text or the reason it was rejected.
    fn llm_text(&mut self, instruction: &str, ids: &[String], book: &FactBook) -> Result<(String, String), String> {
        let name;
        let llm = self.llm.as_mut().ok_or("no language model attached")?;
        let facts: String = ids.iter().filter_map(|i| book.get(i)).take(self.cfg.max_prompt_facts).map(|f| format!("[{}] {}", f.id, clip(&f.text, self.cfg.fact_chars))).collect::<Vec<_>>().join("\n");
        let user = format!("FACTS\n{facts}\n\nTASK\n{instruction}");
        let raw = llm.complete(SYSTEM_TEXT, &user, self.cfg.max_new_tokens)?;
        name = llm.name();
        let text = match parse_reply(&raw) {
            Reply::Answer(a) => a,
            _ => return Err("model reply had no ANSWER: line".into()),
        };
        let rep = check(&text, book);
        if rep.ok { Ok((text, name)) } else { Err(format!("failed grounding: {}", rep.problems.join("; "))) }
    }

    pub fn investigate(&mut self, question: &str, ev: &Evidence, g: &dyn GraphSource) -> Investigation {
        let (mut book, ix) = ev.fact_book();
        let typ = identify(ev);
        let mut executed: Vec<ExecutedQuery> = vec![];
        let mut prov = Provenance::default();
        let mut final_text: Option<String> = None;

        if self.llm.is_some() {
            let mut transcript = String::new();
            let mut nq = 0usize;
            for step in 0..self.cfg.max_steps {
                let user = format!("FACTS\n{}\n\nQUESTION\n{}{}", render_facts(&book, &self.cfg), question, transcript);
                let name;
                let raw = {
                    let llm = self.llm.as_mut().unwrap();
                    let r = llm.complete(SYSTEM, &user, self.cfg.max_new_tokens);
                    name = llm.name();
                    match r {
                        Ok(t) => t,
                        Err(e) => {
                            prov.fallback_reason = Some(format!("language model unavailable: {e}"));
                            break;
                        }
                    }
                };
                prov.llm_steps = step + 1;
                match parse_reply(&raw) {
                    Reply::Queries(lines) => {
                        for l in lines {
                            if nq >= self.cfg.max_queries {
                                break;
                            }
                            nq += 1;
                            let e = self.exec(parse_query(&l), &l, g, &mut book);
                            transcript.push_str(&format!("\nYou asked: {}\nResult: {}", e.query, if e.ok { e.fact_ids.iter().map(|i| format!("[{i}]")).collect::<String>() } else { e.note.clone().unwrap_or_default() }));
                            executed.push(e);
                        }
                    }
                    Reply::Answer(a) => {
                        let rep = check(&a, &book);
                        if rep.ok {
                            prov.model = Some(name);
                            prov.llm_used = true;
                            prov.grounding = rep;
                            final_text = Some(a);
                        } else {
                            prov.fallback_reason = Some(format!("model answer rejected: {}", rep.problems.join("; ")));
                        }
                        break;
                    }
                    Reply::Neither => {
                        prov.fallback_reason = Some("model reply had neither QUERY nor ANSWER lines".into());
                        break;
                    }
                }
                if step + 1 == self.cfg.max_steps && final_text.is_none() && prov.fallback_reason.is_none() {
                    prov.fallback_reason = Some("model gave no answer within the step limit".into());
                }
            }
        }

        let answer = match final_text {
            Some(t) => t,
            None => {
                let have: BTreeSet<String> = executed.iter().map(|e| e.query.clone()).collect();
                let mut new_ids: Vec<String> = executed.iter().flat_map(|e| e.fact_ids.clone()).collect();
                for q in plan(ev, question) {
                    if have.contains(&q.render()) {
                        continue;
                    }
                    let e = self.exec(Ok(q), "", g, &mut book);
                    new_ids.extend(e.fact_ids.clone());
                    executed.push(e);
                }
                let text = self.compose(ev, &book, &ix, &typ, &new_ids);
                prov.grounding = check(&text, &book);
                text
            }
        };
        Investigation { question: question.into(), answer, facts: book.facts, queries: executed, typologies: typ, provenance: prov, disclaimer: DISCLAIMER.into() }
    }

    /// Deterministic answer built only from facts.
    fn compose(&self, ev: &Evidence, book: &FactBook, ix: &FactIndex, typ: &[TypologyHit], query_ids: &[String]) -> String {
        let mut s: Vec<String> = vec![];
        if let Some(a) = ix.alert.first() {
            s.push(format!("{}.", cited(book, a)));
        } else if book.is_empty() && query_ids.is_empty() {
            return "No evidence was available for this account, so nothing can be said about it.".into();
        }
        if !(typ.is_empty() && ix.alert.is_empty()) {
            s.push(typology_sentence(typ, ix));
        }
        if let Some(i) = ix.graph.first() {
            s.push(format!("{}.", cited(book, i)));
        }
        for id in query_ids.iter().take(4) {
            s.push(format!("{}.", cited(book, id)));
        }
        for id in ix.ownership.iter().take(2).chain(ix.sanctions.iter().take(2)).chain(ix.history.iter().take(1)).chain(ix.feedback.iter().take(1)) {
            s.push(format!("{}.", cited(book, id)));
        }
        let _ = ev;
        s.push("These are indicators that warrant review, not findings of wrongdoing.".into());
        s.join(" ")
    }

    pub fn explain_alert(&mut self, ev: &Evidence, g: &dyn GraphSource) -> Explanation {
        let inv = self.investigate("Explain why this alert was raised and what an analyst should check first.", ev, g);
        let (book, ix) = ev.fact_book();
        let mut key: Vec<String> = ix.alert.clone();
        key.extend(ix.rules.iter().map(|r| r.1.clone()));
        key.extend(ix.graph.clone());
        key.extend(ix.ownership.clone());
        key.extend(ix.sanctions.clone());
        key.extend(ix.history.clone());
        key.truncate(8);
        let mut key_facts: Vec<Fact> = key.iter().filter_map(|i| book.get(i).cloned()).collect();
        key_facts.extend(inv.facts.iter().skip(book.len()).take(4).cloned());
        let mut steps: Vec<String> = vec![];
        for t in inv.typologies.iter().take(3) {
            if let Some(d) = typology::def(&t.code) {
                for s in d.next_steps {
                    if !steps.iter().any(|x| x == s) {
                        steps.push((*s).to_string());
                    }
                }
            }
        }
        if ev.notes.is_empty() {
            steps.push("Record the reasoning for the decision in the case notes so the outcome can be audited and learned from.".into());
        }
        steps.truncate(6);
        let headline = match (&ev.alert, inv.typologies.first()) {
            (Some(a), Some(t)) => format!("Alert {} on account {}: {}", a.id, ev.subject, t.name),
            (Some(a), None) => format!("Alert {} on account {}: no known typology matched", a.id, ev.subject),
            _ => format!("Account {}", ev.subject),
        };
        Explanation { alert_id: ev.alert.as_ref().map(|a| a.id), headline, narrative: inv.answer, typologies: inv.typologies, key_facts, next_steps: steps, queries: inv.queries, provenance: inv.provenance, disclaimer: DISCLAIMER.into() }
    }

    pub fn sar_draft(&mut self, case: &CaseFacts, evs: &[Evidence], g: &dyn GraphSource) -> SarDraft {
        let mut book = FactBook::default();
        let mut idx: Vec<FactIndex> = vec![];
        for e in evs {
            idx.push(e.facts_into(&mut book));
        }
        let mut note_ids = vec![];
        for n in &case.notes {
            note_ids.push(book.add(format!("Case note by {} on {}: {}", n.author, n.at, clip(&n.body.split_whitespace().collect::<Vec<_>>().join(" "), 300))));
        }
        // enrich with bounded graph queries per subject
        let mut query_ids: Vec<String> = vec![];
        let mut seen = BTreeSet::new();
        for e in evs {
            if !seen.insert(e.subject) {
                continue;
            }
            for q in plan(e, "").into_iter().take(3) {
                let r = run_query(&q, g);
                query_ids.extend(r.facts.into_iter().map(|t| book.add(t)));
            }
        }
        let typ_all: Vec<Vec<TypologyHit>> = evs.iter().map(identify).collect();
        let section = |title: &str, ids: Vec<String>, lead: Option<String>| -> SarSection {
            let mut body = lead.unwrap_or_default();
            for id in &ids {
                if !body.is_empty() {
                    body.push(' ');
                }
                body.push_str(&format!("{}.", cited(&book, id)));
            }
            if body.is_empty() {
                body = "No information is available for this section.".into();
            }
            SarSection { title: title.into(), body, fact_ids: ids }
        };
        let mut prov = Provenance::default();
        let mut sections = vec![];

        let mut ids: Vec<String> = vec![];
        for ix in &idx {
            ids.extend(ix.account.clone());
            ids.extend(ix.customer.clone());
            ids.extend(ix.ownership.iter().take(1).cloned());
        }
        sections.push(section("1. Subject and accounts", ids, None));

        let mut sum_ids: Vec<String> = vec![];
        for ix in &idx {
            sum_ids.extend(ix.alert.clone());
            sum_ids.extend(ix.rules.iter().map(|r| r.1.clone()).take(4));
            sum_ids.extend(ix.graph.clone());
            sum_ids.extend(ix.sanctions.clone());
        }
        let mut summary = section("2. Summary of suspicion", sum_ids.clone(), None);
        if self.llm.is_some() {
            match self.llm_text("Write three to five sentences summarising why this activity may be suspicious. Cite the facts you use.", &sum_ids, &book) {
                Ok((t, name)) => {
                    summary.body = t;
                    prov.llm_used = true;
                    prov.model = Some(name);
                }
                Err(e) => prov.fallback_reason = Some(format!("summary: {e}")),
            }
        }
        sections.push(summary);

        let mut tx: Vec<String> = vec![];
        for ix in &idx {
            tx.extend(ix.transactions.iter().take(10).cloned());
        }
        tx.extend(query_ids.clone());
        sections.push(section("3. Description of the activity", tx, None));

        let mut ind: Vec<String> = vec![];
        for ix in &idx {
            ind.extend(ix.rules.iter().map(|r| r.1.clone()));
        }
        let names: BTreeSet<String> = typ_all.iter().flatten().map(|t| t.name.clone()).collect();
        let lead = if names.is_empty() { "No known typology matched.".to_string() } else { format!("Typologies indicated: {}.", names.into_iter().collect::<Vec<_>>().join("; ")) };
        sections.push(section("4. Indicators and typologies", ind, Some(lead)));

        let mut own: Vec<String> = vec![];
        for ix in &idx {
            own.extend(ix.ownership.clone());
            own.extend(ix.sanctions.clone());
        }
        sections.push(section("5. Ownership and sanctions screening", own, None));

        let mut hist: Vec<String> = vec![];
        for ix in &idx {
            hist.extend(ix.history.clone());
            hist.extend(ix.notes.clone());
        }
        hist.extend(note_ids.clone());
        sections.push(section("6. Prior history and analyst notes", hist, None));

        sections.push(SarSection {
            title: "7. Decision and filing".into(),
            body: "To be completed by the MLRO: decision, SAR reference, date and channel of submission to the competent authority. The format and mandatory fields must follow that authority's current requirements.".into(),
            fact_ids: vec![],
        });

        let mut open = vec![];
        if evs.iter().all(|e| e.customer.is_none()) {
            open.push("No customer KYC record is linked to the accounts: add identification and source-of-funds information.".to_string());
        }
        if evs.iter().all(|e| e.ownership.is_none()) {
            open.push("No beneficial-ownership data was available: establish who owns and controls the account holder.".to_string());
        }
        if note_ids.is_empty() && evs.iter().all(|e| e.notes.is_empty()) {
            open.push("No analyst notes exist: document the investigation steps and reasoning.".to_string());
        }
        if evs.iter().all(|e| e.transactions.is_empty()) {
            open.push("No individual transactions are attached: list the flagged transactions with dates and amounts.".to_string());
        }
        if typ_all.iter().all(|t| t.is_empty()) {
            open.push("No known typology matched: state the grounds for suspicion in your own words.".to_string());
        }
        open.push("MLRO review, decision and approval are required before anything is filed.".to_string());
        if let Some(r) = &prov.fallback_reason {
            open.push(format!("The language model's text was not used ({r}); the summary was built from the facts."));
        }
        prov.grounding = check(&sections.iter().take(6).map(|s| s.body.clone()).collect::<Vec<_>>().join("\n"), &book);
        SarDraft { case_id: case.case_id, status: "DRAFT - not filed; requires MLRO review and approval".into(), sections, open_items: open, facts: book.facts, provenance: prov, disclaimer: DISCLAIMER.into() }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskSummary {
    pub alerts: usize,
    pub by_typology: Vec<(String, usize)>,
    pub sanctions_related: usize,
    pub text: String,
}

/// Portfolio view of a set of alerts: counts only, computed in code (no model involved).
pub fn risk_summary(evs: &[Evidence]) -> RiskSummary {
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    let mut sanc = 0;
    for e in evs {
        let t = identify(e);
        if t.iter().any(|x| x.code == "sanctions") {
            sanc += 1;
        }
        if let Some(top) = t.first() {
            *counts.entry(top.name.clone()).or_default() += 1;
        }
    }
    let mut v: Vec<(String, usize)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let text = if evs.is_empty() {
        "No open alerts.".to_string()
    } else {
        format!(
            "{} alert(s) in view; {} with sanctions indicators. Leading typologies: {}.",
            evs.len(), sanc, if v.is_empty() { "none identified".into() } else { v.iter().take(3).map(|(n, c)| format!("{n} ({c})")).collect::<Vec<_>>().join(", ") }
        )
    };
    RiskSummary { alerts: evs.len(), by_typology: v, sanctions_related: sanc, text }
}
