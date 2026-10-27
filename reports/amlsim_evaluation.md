# Performance on the AMLSim-style archive (archive__1_.zip)

Data: `transactions.csv` (1,323,234 transfers, 200 day-steps), `alerts.csv` (1,719 labelled rows in 391 alert groups: 192 `cycle`, 199 `fan_in`),
`accounts.csv` (10,000 US individuals, 5 behaviour classes). 1,719 fraud transactions (0.130%). Loaded with the new `amlsim` adapter
(`aml-eval --format amlsim`; one step = one day; amounts treated as USD). `accounts.IS_FRAUD` and `transactions.ALERT_ID` are never used as inputs.

## What the data looks like (and what is an artifact)
* Every fraud transaction has an amount between 2.5 and 20; the 1.32M normal ones have a median of 157. Four of the five account behaviour
  classes normally send 150 - 3.8 million per payment, so a fraud payment is 10x to 100,000x **below the sender's own usual amount**.
  The fifth class (behaviour 1, 29% of traffic) normally sends 5 - 20, the same range as the fraud.
* `fan_in` patterns: 3-5 senders, ~2 days apart, one receiver. `cycle` patterns: 4-5 hops, same amount on every hop, up to ~9 days.
* The graph is dense (1.32M edges over 10k nodes), so loops and fan-ins occur by chance everywhere.
* 3,053 payments carry 21,474,836.47 = INT32_MAX / 100, an overflow sentinel. All accounts are US, all transfers domestic, one transaction type.
* Single-feature check (causal, last 30% held out): log-amount alone has ROC-AUC 0.96; structure alone 0.75. A gradient-boosted model on
  causal amount + relative-amount + structure features reaches AP 0.77 (ceiling estimate used only to judge the rules).

## System before (v0.6, built-in rules), full file, threshold 60
Precision 0.56%, recall 54.3%, 165,756 alerts (12.5% of all transactions), 164,822 false positives. Cycle recall 82.8%, fan-in 20.3%.
Noise sources: CIRCULAR_FLOW 130k hits (5.4x lift), PASS_THROUGH 107k (0.5x), FAN_OUT 81k (0.5x), LARGE_TXN 25k (0 true), FAN_IN 34k (2.1x).
Engine speed 72k rows/s. The graph model, trained for the default 3 epochs on the first 330k rows: test AP 0.10%, AUC 0.21 (not learning).

## Improvements (calibration on the first 70% of days, evaluation on the last 30%: 396,971 rows, 551 fraud)
| step | precision | recall | alerts | cycle recall | fan-in recall |
|---|---|---|---|---|---|
| v0.6 defaults | 0.57% | 50.8% | 49,538 | 81.7% | 16.8% |
| + `AMOUNT_DROP` rule, weights fitted | 3.59% | 79.1% | 12,153 | 77.9% | 80.5% |
| + cycle value conservation + LARGE_TXN relative to sender | 4.93% | 90.6% | 10,130 | 99.7% | 80.5% |
| + fitter scales secondary weights (LARGE_TXN extreme weight) = `profiles/amlsim_fitted.json` | **92.6%** | **90.6%** | **539** | 99.7% | 80.5% |

Reproduce: `aml-eval --csv transactions.csv --format amlsim --config profiles/amlsim_fitted.json --eval-from 0.7`.
After the change 40 false positives and 52 missed fraud payments remain (all missed are fan-in payments from accounts whose normal amounts look like the fraud amounts).

Graph model (51-day slice, 40 epochs, rule features from the fitted profile; the profile saw this slice, so this row is in-sample for the rules):
rules online AP 64.1% / F1 74.4%; retro-linked AP 77.8%; graph model AP 8.9%. Without the profile, 40 epochs give the graph model AP 4.2%
(rules 0.22%). **On this dataset the calibrated rules beat the graph model; the graph model does not add value here.**

## Cautions
* The decisive signal (`AMOUNT_DROP`) exploits how this generator scales amounts per behaviour class. The mechanism (a payment far below the
  sender's own norm) is a recognised probing / structuring indicator, but its weight here (0.68) is dataset-specific: the rule is **off by default**
  and only the AMLSim profile switches it on. Expect lower numbers on real portfolios.
* The 70/30 split is by time inside one generator run. It shows the fitted weights generalise to later days of the same simulation, not to other data.
* Labels are generator labels, not analyst decisions; precision of 92% says nothing about analyst workload on real alerts.

## Through the product (PostgreSQL 16, `aml import-dataset`, profile imported with `POST /rules/import`)
Full file: 1,323,034 transactions imported in 38 s (200 rows refused by API validation: 19 zero amounts and self-transfers), **1,295 alerts / 1,607 flagged
transactions** (alerts by rule: AMOUNT_DROP 953, CIRCULAR_FLOW 182, both 105). The same file with the built-in rules produced 165,756 alerts offline.
(Full-file counts include the 70% the profile was fitted on; the held-out figures above are the honest ones.)
Two defects found by this run and fixed: the rule-import validator rejected the new parameters, and the adapter let zero-amount rows through to the API.
