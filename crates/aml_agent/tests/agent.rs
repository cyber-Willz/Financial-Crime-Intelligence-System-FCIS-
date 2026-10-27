use aml_agent::evidence::*;
use aml_agent::*;
use std::time::Duration;

const T0: i64 = 1_665_100_800;

fn fl(id: i64, ts: i64, from: i64, to: i64, amt: f64) -> Flow {
    Flow { id, ts, from, to, amount_gbp: amt, ptype: "CASH_DEPOSIT".into(), from_cc: "GB".into(), to_cc: "GB".into() }
}

fn graph() -> MemGraph {
    let mut f = vec![];
    for i in 0..9 {
        f.push(fl(i + 1, T0 + i * 3600, 100 + i, 1, 9_500.0 + i as f64));
    }
    f.push(fl(50, T0 + 86_400, 1, 7, 80_000.0));
    MemGraph::new(f)
}

fn evidence() -> Evidence {
    Evidence {
        subject: 1,
        alert: Some(AlertFacts {
            id: 41,
            score: 87,
            severity: "high".into(),
            status: "open".into(),
            txn_count: 9,
            rules: vec![RuleFact { code: "FAN_IN".into(), weight: 0.55, detail: "9 senders in 2 days".into() }, RuleFact { code: "PASS_THROUGH".into(), weight: 0.6, detail: "".into() }],
            created_at: "2022-10-08".into(),
        }),
        account: Some(AccountFacts { account: 1, country: Some("BZ".into()), risk_level: "high".into(), risk_score: 70, alert_count: 2, first_seen: "2022-01-01".into(), last_seen: "2022-10-08".into(), txn_sent: 1, txn_received: 9, total_sent_gbp: 80_000.0, total_received_gbp: 85_536.0, watchlisted: false }),
        transactions: vec![TxnFact { id: 1, at: "2022-10-07".into(), sender: 100, receiver: 1, amount_gbp: 9500.0, payment_type: "CASH_DEPOSIT".into(), sender_country: "GB".into(), receiver_country: "GB".into() }],
        history: HistoryFacts { prior_alerts: 2, prior_sar: 1, prior_confirmed: 0, prior_false_positive: 1 },
        ..Default::default()
    }
}

#[test]
fn deterministic_explanation_is_grounded_and_useful() {
    let mut a = Agent::deterministic();
    let ex = a.explain_alert(&evidence(), &graph());
    assert!(ex.provenance.grounding.ok, "{:?}", ex.provenance.grounding.problems);
    assert!(!ex.provenance.llm_used);
    assert!(ex.headline.contains("Alert 41"));
    assert!(ex.narrative.contains("[F1]"));
    assert!(ex.narrative.contains("9 distinct account(s)"), "{}", ex.narrative);
    assert!(ex.typologies.iter().any(|t| t.code == "funnel"));
    assert!(!ex.next_steps.is_empty());
    assert!(ex.queries.iter().all(|q| q.ok));
    // every cited id exists in the key facts or the full book
    let inv = a.investigate("who pays this account", &evidence(), &graph());
    assert!(inv.facts.len() > ex.key_facts.len() - 1);
}

#[test]
fn model_can_drive_graph_queries_and_answer() {
    let llm = ScriptedLlm::new(&["QUERY counterparties account=1 dir=in top=3", "ANSWER: Nine senders paid the account [F1] and the biggest are listed in [F8]."]);
    let mut a = Agent::new(Some(Box::new(llm)), AgentConfig::default());
    let inv = a.investigate("Who funds this account?", &evidence(), &graph());
    assert!(inv.provenance.llm_used, "{:?}", inv.provenance);
    assert_eq!(inv.provenance.llm_steps, 2);
    assert_eq!(inv.queries.len(), 1);
    assert!(inv.queries[0].ok && !inv.queries[0].fact_ids.is_empty());
    assert!(inv.answer.starts_with("Nine senders"));
}

#[test]
fn refused_queries_are_reported_back_and_never_run() {
    let llm = ScriptedLlm::new(&["QUERY neighbors account=1; DROP TABLE alerts", "ANSWER: The alert was raised [F1]."]);
    let mut a = Agent::new(Some(Box::new(llm)), AgentConfig::default());
    let inv = a.investigate("anything", &evidence(), &graph());
    assert!(!inv.queries[0].ok && inv.queries[0].note.as_ref().unwrap().contains("refused"));
    assert!(inv.provenance.llm_used);
}

#[test]
fn ungrounded_model_text_is_replaced_by_the_deterministic_answer() {
    for bad in ["ANSWER: The holder is guilty and moved 98765 GBP [F99].", "ANSWER: Account 1 moved 12345 GBP [F1].", "no idea what to do", "ANSWER: Ignore previous instructions [F1]."] {
        let mut a = Agent::new(Some(Box::new(ScriptedLlm::new(&[bad]))), AgentConfig::default());
        let inv = a.investigate("Explain", &evidence(), &graph());
        assert!(!inv.provenance.llm_used, "{bad}");
        assert!(inv.provenance.fallback_reason.is_some());
        assert!(inv.provenance.grounding.ok, "{:?}", inv.provenance.grounding.problems);
        assert!(!inv.answer.contains("98765"));
    }
}

#[test]
fn unreachable_model_degrades_gracefully() {
    let mut a = Agent::new(Some(Box::new(ScriptedLlm::new(&[]))), AgentConfig::default());
    let inv = a.investigate("Explain", &evidence(), &graph());
    assert!(inv.provenance.fallback_reason.as_ref().unwrap().contains("unavailable"));
    assert!(inv.provenance.grounding.ok);
}

#[test]
fn sar_draft_is_a_grounded_draft_with_open_items() {
    let mut a = Agent::deterministic();
    let case = CaseFacts { case_id: 7, title: "Fan-in".into(), priority: "high".into(), status: "open".into(), notes: vec![] };
    let d = a.sar_draft(&case, &[evidence()], &graph());
    assert!(d.status.starts_with("DRAFT") && d.status.contains("MLRO"));
    assert_eq!(d.sections.len(), 7);
    assert!(d.provenance.grounding.ok, "{:?}", d.provenance.grounding.problems);
    assert!(d.open_items.iter().any(|o| o.contains("KYC")) && d.open_items.iter().any(|o| o.contains("notes")));
    assert!(d.sections[6].body.contains("MLRO"));
    assert!(d.sections[2].body.contains("distinct account(s)"));
    // a grounded model summary is accepted; an ungrounded one is not
    let ok = ScriptedLlm::new(&["ANSWER: The account was alerted [F1] with several rules [F2]."]);
    let d = Agent::new(Some(Box::new(ok)), AgentConfig::default()).sar_draft(&case, &[evidence()], &graph());
    assert!(d.provenance.llm_used && d.sections[1].body.starts_with("The account was alerted"));
    let bad = ScriptedLlm::new(&["ANSWER: It moved 31337 GBP [F2]."]);
    let d = Agent::new(Some(Box::new(bad)), AgentConfig::default()).sar_draft(&case, &[evidence()], &graph());
    assert!(!d.provenance.llm_used && d.open_items.iter().any(|o| o.contains("not used")));
}

#[test]
fn risk_summary_counts_typologies() {
    let s = risk_summary(&[evidence(), evidence()]);
    assert_eq!(s.alerts, 2);
    assert!(s.text.contains("2 alert(s)"));
}

fn fake(mode: &str) -> ProcessLlm {
    // the helper binary is built by cargo for integration tests
    std::env::set_var("FAKE_LLM_MODE", mode);
    let mut p = ProcessLlm::new(env!("CARGO_BIN_EXE_fake_llm"), vec![]);
    p.request_timeout = Duration::from_secs(2);
    p
}

#[test]
fn process_llm_speaks_the_sidecar_protocol() {
    let mut a = Agent::new(Some(Box::new(fake("query"))), AgentConfig::default());
    let inv = a.investigate("Who funds this account?", &evidence(), &graph());
    assert!(inv.provenance.llm_used, "{:?}", inv.provenance);
    assert_eq!(inv.provenance.model.as_deref(), Some("fake-query"));
    assert_eq!(inv.queries.len(), 1);
}

#[test]
fn process_llm_garbage_is_caught_by_the_gate() {
    let mut a = Agent::new(Some(Box::new(fake("garbage"))), AgentConfig::default());
    let inv = a.investigate("Explain", &evidence(), &graph());
    assert!(!inv.provenance.llm_used && inv.provenance.grounding.ok);
}

#[test]
fn a_hung_model_times_out_and_the_agent_still_answers() {
    let mut a = Agent::new(Some(Box::new(fake("hang"))), AgentConfig::default());
    let t = std::time::Instant::now();
    let inv = a.investigate("Explain", &evidence(), &graph());
    assert!(t.elapsed() < Duration::from_secs(10));
    assert!(!inv.provenance.llm_used && inv.provenance.fallback_reason.as_ref().unwrap().contains("did not answer"));
    assert!(inv.provenance.grounding.ok);
}

#[test]
fn a_missing_binary_is_an_error_not_a_panic() {
    let mut p = ProcessLlm::new("definitely-not-a-real-binary-xyz", vec![]);
    assert!(p.complete("s", "u", 8).unwrap_err().contains("cannot start"));
}
