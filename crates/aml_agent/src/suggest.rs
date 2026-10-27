//! Dynamic rule suggestions: the feedback loop's findings, phrased for a compliance officer.
//! Every suggestion carries the evidence behind it and, where it is a change the system can make, the exact
//! change ([`Apply`]); applying is a separate, audited administrator action.

use aml_feedback::{Action, RuleReport, ThresholdAdvice};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Apply {
    RuleWeight { rule: String, weight: f64 },
    AlertThreshold { value: u8 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Suggestion {
    pub kind: String,
    pub title: String,
    pub detail: String,
    pub apply: Option<Apply>,
}

pub fn suggestions(rules: &RuleReport, thr: &ThresholdAdvice) -> Vec<Suggestion> {
    let mut out = vec![];
    for r in &rules.recommendations {
        match (r.action, r.current_weight, r.suggested_weight) {
            (Action::Increase | Action::Decrease, Some(w0), Some(w1)) => out.push(Suggestion {
                kind: "rule_weight".into(),
                title: format!("{} {} from {:.2} to {:.2}", if r.action == Action::Increase { "Raise" } else { "Lower" }, r.rule, w0, w1),
                detail: r.rationale.clone(),
                apply: Some(Apply::RuleWeight { rule: r.rule.clone(), weight: (w1 * 1000.0).round() / 1000.0 }),
            }),
            _ => {}
        }
    }
    if thr.changed {
        out.push(Suggestion {
            kind: "threshold".into(),
            title: format!("Move the alert threshold from {} to {}", thr.current, thr.recommended),
            detail: thr.rationale.clone(),
            apply: Some(Apply::AlertThreshold { value: thr.recommended }),
        });
    }
    if rules.uncovered_confirmed > 0 {
        out.push(Suggestion {
            kind: "coverage".into(),
            title: format!("{} confirmed alert(s) were caught only by context rules", rules.uncovered_confirmed),
            detail: "Review those cases for a typology the rule set does not model yet; a new rule is a design decision and is not applied automatically.".into(),
            apply: None,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use aml_feedback::rules::{RuleRecommendation, RuleStat};

    fn stat() -> RuleStat {
        RuleStat { rule: "FAN_IN".into(), alerts: 80, confirmed: 60, false_positives: 20, precision: 0.7, ci_low: 0.6, ci_high: 0.8, lift: 2.0, coef: 1.0 }
    }

    #[test]
    fn only_actionable_findings_become_suggestions() {
        let mut rep = RuleReport::default();
        rep.recommendations.push(RuleRecommendation { rule: "FAN_IN".into(), action: Action::Increase, current_weight: Some(0.4), suggested_weight: Some(0.5234), stat: stat(), rationale: "why".into() });
        rep.recommendations.push(RuleRecommendation { rule: "X".into(), action: Action::Keep, current_weight: Some(0.4), suggested_weight: None, stat: stat(), rationale: "".into() });
        rep.uncovered_confirmed = 2;
        let thr = ThresholdAdvice { current: 60, recommended: 65, changed: true, observed_floor: Some(60), table: vec![], calibrator: None, rationale: "r".into(), caveats: vec![] };
        let s = suggestions(&rep, &thr);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].apply, Some(Apply::RuleWeight { rule: "FAN_IN".into(), weight: 0.523 }));
        assert_eq!(s[1].apply, Some(Apply::AlertThreshold { value: 65 }));
        assert!(s[2].apply.is_none());
    }
}
