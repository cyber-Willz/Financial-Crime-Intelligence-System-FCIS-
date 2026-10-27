# AML Transaction Monitoring System

Rust · Axum 0.8 · SeaORM 1.1 · PostgreSQL 16. Scores every payment in real time against a
configurable typology rule set, raises de-duplicated alerts, and carries them through an
analyst → case → SAR workflow with an immutable audit trail.

```
crates/
  aml_datasets   adapters for SAML-D, AML_Suspicious, FraudShield, Risk_prediction -> canonical transactions
  aml_engine     streaming detection engine (no DB, no async) — rules, scoring, rolling windows
  aml_eval       `aml-eval`: back-test the engine on the labelled SAML-D dataset
  aml_entity     SeaORM entities
  aml_migration  SeaORM migration (schema, constraints, indexes, append-only audit trigger)
  aml_api        `aml` binary: Axum API + CLI (migrate | serve | import-saml)
  aml_graph      `aml-gnn`: connectome-informed recurrent graph neural network (v0.4, see crates/aml_graph/README.md)
  aml_bo         `aml-bo`: beneficial-ownership analysis on the CDD/sanctions engine (v0.5, see crates/aml_bo/README.md)
  aml_feedback   (v0.6) connectome memory feedback loop: reinforcement signals -> rule recommendations, threshold calibration, outcome memory, guarded graph-head tuning
  aml_agent      (v0.6) bidirectional LLM analyst agent: evidence -> grounded explanations, typologies, constrained graph queries, prioritisation, SAR drafting
llm_sidecar/     (v0.6) `aml-llm-sidecar`: burn-llm wrapped as a JSON-lines process; its own Cargo workspace (Burn needs rustc >= 1.92)
vendor/          human_connectome_rnn + krylov_ds, spectral_hypergraph, nbsc (path dependencies of aml_graph); cdd_sanctions (aml_bo); burn-llm (sidecar)
```

## v0.6 — the design-guide architecture

```
DATA SOURCES -> INGESTION (aml_datasets, canonical txn, KYC/ownership resolution)
   -> REAL-TIME ENGINE (aml_engine, noisy-OR) -> alerts
        |-> BENEFICIAL OWNERSHIP (aml_bo)           |-> CASE MANAGEMENT (alerts, cases, notes, SAR, audit)
   -> GRAPH LAYER (aml_graph: hypergraph + connectome-RNN memory + graph NN -> graph risk)
   -> BIDIRECTIONAL LLM ANALYST AGENT (aml_agent)  <-- graph queries / investigation tasks -->
   -> CONNECTOME MEMORY FEEDBACK LOOP (aml_feedback): decision -> outcome -> SAR? -> false positive? -> reinforcement
        -> rule recommendations | calibrated threshold | outcome memory | graph-head fine-tune
```

| Design-guide box | Where it lives | Status |
|---|---|---|
| Ingestion, real-time engine, ownership, case management, graph layer | v0.1-v0.5 crates | unchanged |
| Graph risk into the workflow | `aml graph-rescore --model m.json` writes `graph_risk`; used by the agent and the queue | new, batch (end-of-window) |
| Alert explanation, typology identification, risk summarisation | `POST /agent/alerts/{id}/explain`, `GET /agent/summary` | new |
| Natural-language investigation + graph exploration control | `POST /agent/investigate` (closed query grammar, read-only, bounded) | new |
| SAR / regulatory narrative drafting | `POST /agent/cases/{id}/sar-draft` (always a DRAFT for MLRO review) | new |
| Case prioritisation | `GET /alerts/queue` (calibrated probability x graph risk x exposure x age x outcome memory) | new |
| Dynamic rule suggestions | `GET /feedback/summary`, `POST /feedback/recompute`, `.../recommendations/{id}/apply|reject` (admin, audited) | new |
| Calibrate risk thresholds | `aml_feedback::calibrate` (isotonic + Wilson bound, capped step) | new |
| Update graph attention | `aml tune-graph --model m.json --out tuned.json` (head only, held-out guard) | new |
| Update memory states | outcome memory for ordering the queue; RNN hidden states are deliberately not edited (see below) | new, adapted |

### Agent: what the language model may and may not do
The agent works with **no model at all** (every function returns grounded text built from numbered facts).
A local model is optional: `AML_LLM_CMD=<path to aml-llm-sidecar>` and `AML_LLM_MODEL=<model dir>` (optional
`AML_LLM_TIMEOUT_SECS`). It may choose graph queries (`QUERY cycles account=123 ...`) and phrase the answer. Everything it
writes passes the grounding gate: every `[F#]` citation must exist, every number must occur in a fact, no
accusations, no injection residue. Otherwise the deterministic text is returned and `provenance.fallback_reason`
says why. `?deterministic=true` skips the model. Every output is stored in `agent_runs` (evidence digest, model,
grounding result). The agent never changes an alert, case, rule or threshold.

Build the sidecar separately (own lockfile, rustc >= 1.92): `cargo build --release --manifest-path llm_sidecar/Cargo.toml`.
Model directory = `config.json` + safetensors + `tokenizer.json` (Llama / Qwen families, see `vendor/burn-llm`).
Optional ownership evidence: `AML_OWNERSHIP_FILE` = JSON `{"<account>": OwnershipFacts}` (shape in `aml_agent::evidence`).

### Feedback loop: what it learns and how it is kept honest
* `feedback_events` is written automatically when an alert is closed/confirmed/SAR-filed (directly or through case closure)
  and removed if an admin reopens it. Nothing else is stored about analysts' reasoning beyond what they already write.
* **Rules**: marginal precision (shrunk, Wilson interval) *and* a ridge-logistic conditional effect; a rule is recommended
  up/down only if both agree, with >= 30 reviewed alerts, a bounded logit step, and sanctions/watchlist rules never lowered.
* **Threshold**: lowest threshold whose precision lower bound reaches the target (default 25%) while keeping >= 70% of
  confirmed alerts, max +10 points per cycle. Lowering is never recommended (no outcomes exist below the threshold).
* **Outcome memory**: decayed per-account / per-(account, rule) history reorders the queue (0.6x-1.5x). It never
  suppresses or closes an alert; hard controls and SAR history are never down-weighted.
* **Graph head**: only the pairwise head moves, anchored to the old weights; kept only if average precision on the most
  recent 30% of windows improves; otherwise the original model is restored. RNN states stay valid because the RNN is untouched.
* All statistics are conditional on "was alerted and reviewed": the loop cannot find missing rules or undetected laundering,
  and inherits reviewer error. Recommendations are advisory until an admin applies them.

### New environment variables
`AML_LLM_CMD`, `AML_LLM_MODEL`, `AML_LLM_TIMEOUT_SECS`, `AML_OWNERSHIP_FILE`.

### Verification performed for v0.6
* `cargo test --workspace`: unit + integration tests, including `aml_feedback` (20) and `aml_agent` (34: grounding, query
  grammar fuzz cases, scripted and out-of-process models, hung-model timeout).
* Live run on PostgreSQL 16: migration, ingest of synthetic traffic with planted typologies, 243 simulated analyst reviews,
  rule recommendation (CIRCULAR_FLOW 0.80 -> 0.687 because 6 of 208 alerts were real), apply + audit + double-apply refusal,
  calibrated queue, SAR draft, case-close capture, reopen retraction, `graph-rescore`, `tune-graph` (refused: too few holdout positives).
* The real burn-llm sidecar was run against the repository's tiny random-weight fixture model: the protocol works and the
  API falls back cleanly (the fixture's 64-token context cannot hold a prompt). **No pretrained checkpoint was available, so the
  quality of a real model's queries and prose is untested.** The synthetic traffic and simulated analysts prove the plumbing
  and the statistics, not performance on real cases.

## v0.7 — measured on the AMLSim-style archive
Details and the full table: `reports/amlsim_evaluation.md`. On the last 30% of days (calibrated on the first 70%), alerts at threshold 60 went from
49,538 (precision 0.57%, recall 50.8%) to 539 (precision 92.6%, recall 90.6%).
* New dataset adapter `amlsim` (`aml-eval|aml-gnn|aml import-dataset --format amlsim`; `alerts.csv` beside `transactions.csv` supplies typology names).
* New rule `AMOUNT_DROP` (payment far below the sender's own usual amount; **off by default**), `CIRCULAR_FLOW.amount_tol` (every hop must carry about
  the triggering amount; 0 = off), `LARGE_TXN.min_vs_mean` (ignore senders whose normal payments are that large; 0 = off). Defaults are unchanged,
  so existing behaviour and SAML-D calibration are untouched.
* `profiles/amlsim_fitted.json`: calibrated engine config (import with `POST /api/v1/rules/import`, or `aml-eval --config`).
* `aml-eval --enable RULE`; the calibrator now moves a rule's secondary weights (e.g. `LARGE_TXN.extreme_weight`) together with its main weight.
* `aml-gnn eval|score --engine-config profile.json` so the graph model's rule features follow the calibrated profile.
* Finding: on this dataset the calibrated rules outperform the graph model; the signal is amount-relative-to-sender, an artifact of the generator.

## v0.8 — second pass on the AMLSim-style archive
Holdout (calibrated on the first 70% of days, scored on the last 30%, 396,965 rows, 551 fraud), threshold 60:
v0.7 precision 92.58% / recall 90.56% / F1 0.916 (40 FP, 52 FN) -> **v0.8 precision 99.63% / recall 98.73% / F1 0.992 (2 FP, 7 FN)**. Details: `reports/amlsim_evaluation_v08.md`.
* `CIRCULAR_FLOW.repeat_tol` (new, **off by default**): a loop whose closing payment repeats an earlier payment on the same sender->receiver pair with the same value
  (relative tolerance) is a standing/recurring flow, not new round-tripping. Removed 38 of the 40 false positives (one 6-account ring paying identical amounts every 10 days).
* `AMOUNT_DROP.ratio` 0.1 -> 0.6 in `profiles/amlsim_fitted.json`, chosen on the first 70% only (F1 plateau 0.6-0.8; 0.6 is before false positives start to rise). Default unchanged (off).
* `aml-eval --dump-errors errors.csv`: every false positive / false negative in the evaluation window with the rules that fired.
* Defaults and SAML-D behaviour are untouched (v0.6 default-config numbers reproduce exactly: 0.57% / 50.8% on this file).

## Quick start

```powershell
docker compose up -d db                       # or any PostgreSQL 14+
$env:DATABASE_URL = "postgres://aml:aml_dev_pw@localhost:5432/aml"
$env:AML_API_KEYS = "adminkey-0123456789:admin:mlro,analystkey-0123456789:analyst:bob,ingestkey-0123456789:ingest:core"

cargo run --release -p aml_api -- migrate
cargo run --release -p aml_api -- import-dataset --format saml-d --file C:\data\SAML-D.csv --limit 600000   # optional demo data
cargo run --release -p aml_api -- import-dataset --format aml-suspicious --file C:\data\AML_Suspicious_Transactions.csv   # + 5,000 KYC profiles
cargo run --release -p aml_api -- import-dataset --format kyc-risk --file C:\data\transactions_with_fatf_ofac.csv   # + 2,000 KYC clients, sanctioned ones watchlisted
# formats: saml-d | aml-suspicious | kyc-risk | lending | fraudshield | risk-prediction
cargo run --release -p aml_api -- serve        # http://0.0.0.0:8080

Expand-Archive C:\Users\cyber-Willz\Downloads\transactions_csv.zip -DestinationPath C:\data\amlsim
cargo run --release -p aml_api -- import-dataset --format amlsim --file C:\data\amlsim\transactions.csv

cargo run --release -p aml_api -- serve

The rule import never ran, though. serve runs in the foreground and
keeps that PowerShell window busy, so the Invoke-RestMethod line 
after it never got a turn. Open a second PowerShell window and run this:

Invoke-RestMethod -Method Post -Uri http://localhost:8080/api/v1/rules/import `
  -Headers @{ "x-api-key" = "adminkey-0123456789" } `
  -ContentType "application/json" -InFile profiles\amlsim_fitted.json

  alert_threshold applied                                                                    
--------------- -------
             60      21


To see the tuned results, re-import with the profile active.
In the second window, stop serve first with Ctrl+C in
the first window, then:

cargo run --release -p aml_api -- migrate --fresh
cargo run --release -p aml_api -- serve --migrate


Second window: apply the tuned profile.

$env:AML_API_KEYS = "adminkey-0123456789:admin:mlro,analystkey-0123456789:analyst:bob,ingestkey-0123456789:ingest:core"
$env:DATABASE_URL = "postgres://aml:aml_dev_pw@localhost:5432/aml"

Invoke-RestMethod -Method Post -Uri http://localhost:8080/api/v1/rules/import `
  -Headers @{ "x-api-key" = "adminkey-0123456789" } `
  -ContentType "application/json" -InFile profiles\amlsim_fitted.json

First window: press Ctrl+C to stop serve. Then import and restart:

cargo run --release -p aml_api -- import-dataset --format amlsim --file C:\data\amlsim\transactions.csv
cargo run --release -p aml_api -- serve --hydrate-days 400

Then run this in the second window to count the alerts:

(Invoke-RestMethod "http://localhost:8080/api/v1/alerts?limit=1" -Headers @{ "x-api-key" = "adminkey-0123456789" }) | ConvertTo-Json -Depth 3

cargo run --release -p aml_eval -- --csv C:\data\amlsim\transactions.csv --format amlsim --config profiles\amlsim_fitted.json --eval-from 0.7

    Finished `release` profile [optimized] target(s) in 13.06s
     Running `target\release\aml-eval.exe --csv C:\data\amlsim\transactions.csv --format amlsim --config profiles\amlsim_fitted.json --eval-from 0.7`
  .. 1000000 rows, 2s
format=amlsim customers=0
rows=1323215 scored=396965 after 0d warm-up (unmappable 19) laundering=551 base_rate=0.1388%  engine=2s (593061 rows/s)  accounts_tracked=9999

threshold sweep (transaction level)
  thr        TP        FP    alerts     prec   recall       F1
   20       544      9725     10269    5.30%   98.73%    0.101
   30       544         2       546   99.63%   98.73%    0.992
   40       544         2       546   99.63%   98.73%    0.992
   45       544         2       546   99.63%   98.73%    0.992
   50       544         2       546   99.63%   98.73%    0.992
   55       544         2       546   99.63%   98.73%    0.992
   60       544         2       546   99.63%   98.73%    0.992
   70       252         0       252  100.00%   45.74%    0.628
   80       241         0       241  100.00%   43.74%    0.609
   90        12         0        12  100.00%    2.18%    0.043

== at configured alert threshold 60: TP=544 FP=2 precision=99.63% recall=98.73% F1=0.992

recall by laundering typology (flagged at threshold 60)
  fraud                     551 txns  recall   98.7%

per-rule behaviour (direct or linked hits)   [lift = precision / base rate]
  rule                          hits    laund.     prec   recall    lift
  AMOUNT_DROP                    510       509   99.80%   92.38%  719.0x
  BEHAVIOUR_CHANGE              9734        11    0.11%    2.00%    0.8x
  CASH_VELOCITY                    0
  CIRCULAR_FLOW                  277       276   99.64%   50.09%  717.8x
  CURRENCY_CONVERSION              0
  CUSTOMER_JURISDICTION            0
  DORMANT_REACTIVATION             0
  FAN_IN                           0
  FAN_OUT                          0
  HIGH_RISK_CUSTOMER               0
  HIGH_RISK_JURISDICTION           0
  INCOME_MISMATCH                  0
  LARGE_TXN                     9779         0    0.00%    0.00%    0.0x
  LAYERING_CHAIN                 334         0    0.00%    0.00%    0.0x
  PASS_THROUGH                 31936        23    0.07%    4.17%    0.5x
  REPEATED_PAIR                    0
  SANCTIONED_JURISDICTION          0
  STRUCTURING_BAND                 0
  STRUCTURING_HUB                  0
  TRADE_MISPRICING                 0
  WATCHLIST_HIT                    0
```
`scripts/run_e2e.ps1` does all of the above plus a smoke test. For the AMLSim archive use `scripts\run_amlsim.ps1 -Csv C:\data\transactions.csv` (back-test, migrate, apply `profiles/amlsim_fitted.json`, import, serve; `-SkipDb` for the back-test only).

## How detection works

Each transaction is evaluated against per-account rolling windows (≤256 flows/direction, up to
180 days) held in memory, then persisted. Rule weights combine by noisy-OR into a 0-100 score;
score ≥ `alert_threshold` (default 60) raises an alert. Alerts are de-duplicated per
*subject account × day × primary rule* so one pattern is one alert, not hundreds.

| Rule | Typology |
|---|---|
| `WATCHLIST_HIT` | counterparty on internal/sanctions watchlist |
| `LARGE_TXN` | single large / over-invoicing-scale value |
| `HIGH_RISK_JURISDICTION`, `CURRENCY_CONVERSION` | geography / FX context |
| `FAN_IN`, `FAN_OUT` | many counterparties in a short window, **relative to the account's own history** (`min_focus`) |
| `STRUCTURING_HUB` | slow fan-in of sub-threshold amounts from many senders |
| `PASS_THROUGH`, `LAYERING_CHAIN` | funds forwarded almost in full within hours; chains of such hops |
| `CIRCULAR_FLOW` | round-tripping through a 3-5 hop loop |
| `REPEATED_PAIR`, `CASH_VELOCITY` | smurfing / drip cash movements |
| `BEHAVIOUR_CHANGE`, `DORMANT_REACTIVATION` | deviation from the account's own pattern |
| `HIGH_RISK_CUSTOMER`, `CUSTOMER_JURISDICTION` | KYC rating / PEP; high-risk customer **in** a very-high-risk jurisdiction (62% suspicious in the benchmark vs 3% for either alone) |
| `SANCTIONED_JURISDICTION` | counterparty in a sanctioned jurisdiction (alerts alone); sender's own country counts at half weight |
| `STRUCTURING_BAND` | payment within 10% below the reporting threshold **and** out of pattern for the sender; repeats within 7 days strengthen it |
| `TRADE_MISPRICING` | large payment (>= 50k) from an import/export customer |
| `INCOME_MISMATCH` | single amount ≥ 50% of declared annual income, or 30-day outflow ≥ 100% (unvalidated on labelled AML data) |

When a pattern completes (e.g. the 6th sender of a fan-in), the *earlier legs* are linked
retroactively and pick up that rule's weight. All rules, weights, parameters, the alert threshold
and jurisdiction lists are editable at runtime (`PUT /api/v1/rules/{code}`, `/settings`,
`/jurisdictions/{cc}`); changes apply to subsequently scored transactions and are audited.

## API (`/api/v1`, header `x-api-key`)

| Method & path | Role |
|---|---|
| `POST /transactions`, `POST /transactions/batch` (≤5000) | ingest / admin |
| `GET /transactions`, `/transactions/{id}`, `/accounts`, `/accounts/{n}`, `/accounts/{n}/transactions`, `/accounts/{n}/network` | any |
| `GET /alerts`, `/alerts/{id}`; `PATCH /alerts/{id}` | read: any · update: analyst/admin |
| `POST /cases`, `GET /cases[/{id}]`, `PATCH /cases/{id}`, `POST /cases/{id}/notes`, `/alerts` | analyst/admin |
| `POST /cases/{id}/close` (`no_action` or `sar_filed` + `sar_reference`) | analyst · **sar_filed: admin** |
| `PUT /customers/{ref}` (KYC rating, PEP, income, linked accounts), `GET /customers[/{ref}]` | analyst/admin · read: any |
| `POST /rules/import` (apply an `aml-eval --fit-out` calibration, all-or-nothing) | admin |
| `GET /rules`, `PUT /rules/{code}`, `PUT /settings`, `PUT /jurisdictions/{cc}`, `POST/DELETE /watchlist` | admin |
| `GET /stats/overview`, `GET /audit`, `GET /health` (no auth) | any |

Alert workflow: `open → under_review → escalated → sar_filed | closed_confirmed | closed_false_positive`;
closing requires a resolution note; reopening a closed alert is admin-only.
Transaction body: `sender_account, receiver_account, amount, payment_currency, received_currency,
sender_country (ISO-2), receiver_country, payment_type, [occurred_at], [amount_gbp]`.

## Measured performance (SAML-D, 9.5M labelled transactions)

Back-test with `aml-eval --csv SAML-D.csv` (first 14 days treated as warm-up; 9.1M scored,
9,393 laundering transactions, base rate 0.10%). Engine speed ≈ 180k tx/s; full pipeline with
PostgreSQL persistence ≈ 16k tx/s on a single shared core.

| Alert threshold | Precision | Recall | Alerts raised |
|---|---|---|---|
| 50 | 1.2% | 32% | 262,843 |
| **60 (default)** | **8.6%** | **18.8%** | **20,509** |
| 70 | 10.4% | 8.8% | 7,929 |

Rule lift vs. base rate: STRUCTURING_HUB ≈ 250×, CIRCULAR_FLOW / LAYERING_CHAIN ≈ 800-1000×, LARGE_TXN 9×,
FAN_IN 7×, REPEATED_PAIR 7×. Recall by typology at threshold 50: Structuring 78%, Fan_In 77%,
Over-Invoicing 100%, Single_large 47%, Smurfing 45%, Cash_Withdrawal 43%, Cycle 30%; **weak:** Deposit-Send
(3%), Layered_Fan_Out (0.4%), Bipartite (1%), Behavioural_Change (0%).

**Read these numbers honestly.** SAML-D is synthetic, and its normal traffic deliberately contains
heavy fan-in/out, so naive count rules do worse here than on real portfolios; conversely real data
has behaviours the generator lacks. Transaction-level precision is low by nature at a 0.1% base
rate — alert de-duplication and the analyst workflow are what make volume workable. Treat the
default weights as a calibrated starting point, re-tune on your own labelled history, and
validate against your regulator's expectations before relying on it for compliance.

## v0.4: graph-memory layer

`aml_graph` adds an end-of-window model (transaction graph → temporal hypergraph → connectome-RNN memory → GNN → risk) that sees multi-day
structure the single-pass rules cannot. It is additive: `aml_engine` and the API are unchanged. Evidence so far is on synthetic data only and
the connectome wiring itself did not beat random wiring; read `crates/aml_graph/README.md` before relying on it.

## v0.5: beneficial ownership

`aml_bo` plugs the vendored `cdd_sanctions` engine (KYC register, ownership percentages, sanctions screening with a four-eyes
alert workflow, spectral network exposure) into transaction monitoring. It resolves ultimate beneficial owners per account
holder, scores ownership opacity, applies the 50% rule, builds ownership groups (accounts that belong to one customer),
runs ownership-aware monitoring rules, and feeds 12 node and 4 edge ownership features to the `aml_graph` model.
On the synthetic world the ownership rules lift the rule engine a lot, but they added **no measurable gain to the graph model**
(3 seeds); read `crates/aml_bo/README.md` before relying on it.

## Using the extra datasets (v0.2 and v0.3)

v0.3 adds the sanctioned-jurisdiction tier, three rules, sector/PEP/sanctions KYC attributes and the `kyc-risk` / `lending` adapters; see the
"Round 2" section of **DATASETS.md** (including two regressions the data exposed and how they were fixed). Upgrade with `aml migrate`.

### v0.2 notes

See **DATASETS.md** for what each of the nine datasets showed. Headlines: only AML_Suspicious carried real signal (jurisdiction ×
KYC rating), which produced the customer layer and the `very_high` jurisdiction tier; FraudShield and Risk_prediction are noise and are
used as false-positive controls (0 alerts); credit datasets only sanity-checked the income rule.
Calibrate to your own data: `aml-eval --format aml-suspicious --csv X --fit-out tuned.json --eval-from 0.7`
(fit on the first 70%, report on the unseen last 30%), then `POST /api/v1/rules/import`.
On the AML_Suspicious holdout this moved F1 from 0.49 to 0.72.
Upgrading from v0.1: `aml migrate` (adds `customers`, `accounts.customer_id`, 4 payment types, the `very_high` tier) preserves existing data.

## Design notes & known limits

* Ingestion is serialised under one engine lock (score + persist are atomic w.r.t. state). If a
  DB write fails the engine is rebuilt from the database, so memory never runs ahead of storage.
  Horizontal scale-out would need account-sharded engines.
* On start the engine rehydrates from the last `AML_HYDRATE_DAYS` of stored transactions.
* The FX table in `default_config()` is a **placeholder**; supply `amount_gbp` or real rates.
* Default high-risk jurisdictions are a seed derived from the benchmark data, **not** an official
  FATF list — replace them with your own risk assessment.
* Time-ordered input is assumed within a window (batches are sorted internally); heavily
  back-dated events degrade window accuracy.
* Out of scope: sanctions-list name screening, KYC onboarding workflow (ratings are stored and used, not derived), SAR document generation
  and regulator submission, rescoring history after rule changes, TLS (terminate at a proxy).


┌───────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                           │
├───────────────────────────────────────────────────────────────┤
│ Payments │ SWIFT │ ACH │ KYC │ UBO │ Watchlists │ Sanctions  │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                            │
├───────────────────────────────────────────────────────────────┤
│ aml_datasets                                                  │
│ Canonical Transactions                                        │
│ KYC Resolution                                                │
│ Ownership Resolution                                          │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                 REAL-TIME AML ENGINE                          │
├───────────────────────────────────────────────────────────────┤
│ aml_engine                                                    │
│                                                               │
│ • Structuring Hub                                             │
│ • Fan-In / Fan-Out                                            │
│ • Layering Chain                                              │
│ • Circular Flow                                               │
│ • Jurisdiction Risk                                           │
│ • KYC Risk                                                    │
│ • Income Mismatch                                             │
│ • Watchlist Hits                                              │
│                                                               │
│ Noisy-OR Risk Score → Alert                                  │
└───────────────┬──────────────────────────────┬───────────────┘
                │                              │
                ▼                              ▼

┌──────────────────────┐       ┌────────────────────────────┐
│ BENEFICIAL OWNERSHIP │       │ CASE MANAGEMENT            │
├──────────────────────┤       ├────────────────────────────┤
│ aml_bo               │       │ Alerts                     │
│                      │       │ Cases                      │
│ UBO Resolution       │       │ Notes                      │
│ 50% Rule             │       │ Escalations                │
│ Ownership Groups     │       │ SAR Workflow               │
│ Opacity Scoring      │       │ Audit Trail                │
└──────────┬───────────┘       └───────────┬────────────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼

┌───────────────────────────────────────────────────────────────┐
│                  GRAPH INTELLIGENCE                           │
├───────────────────────────────────────────────────────────────┤
│ aml_graph                                                     │
│                                                               │
│ Transaction Graph                                             │
│          ↓                                                    │
│ Temporal Hypergraph                                           │
│          ↓                                                    │
│ Connectome-RNN Memory                                         │
│          ↓                                                    │
│ Graph Neural Network                                          │
│          ↓                                                    │
│ Graph Risk                                                    │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼

┌───────────────────────────────────────────────────────────────┐
│                LLM ANALYST COPILOT (aml_agent)                │
├───────────────────────────────────────────────────────────────┤
│ Inputs                                                        │
│ • Alerts                                                      │
│ • Graph Risk                                                  │
│ • Ownership Evidence                                          │
│ • KYC Information                                              │
│ • Case Notes                                                  │
│                                                               │
│ Functions                                                     │
│ • Explain Alerts                                              │
│ • Identify Typologies                                         │
│ • Graph Investigation                                         │
│ • Queue Prioritisation                                        │
│ • SAR Draft Generation                                        │
│                                                               │
│ HARD CONSTRAINTS                                              │
│ ✗ Cannot change alerts                                        │
│ ✗ Cannot change rules                                         │
│ ✗ Cannot change thresholds                                    │
│ ✗ Cannot close cases                                          │
└──────────────────────────────┬────────────────────────────────┘
                               │
                               ▼

                    ┌─────────────────┐
                    │ Human Analyst   │
                    │     MLRO        │
                    └────────┬────────┘
                             │
                             ▼

┌───────────────────────────────────────────────────────────────┐
│          CONNECTOME FEEDBACK LOOP (aml_feedback)              │
├───────────────────────────────────────────────────────────────┤
│ Review Outcome                                                │
│        ↓                                                      │
│ SAR Filed?                                                    │
│ False Positive?                                               │
│ Confirmed?                                                    │
│        ↓                                                      │
│ • Rule Recommendations                                        │
│ • Threshold Calibration                                       │
│ • Outcome Memory                                              │
│ • Queue Reordering                                            │
│ • Guarded Graph-Head Tuning                                   │
│                                                               │
│ HARD CONSTRAINTS                                              │
│ ✗ No automatic deployment                                     │
│ ✗ No alert suppression                                        │
│ ✗ No RNN state editing                                        │
│ ✗ No lowering sanctions controls                              │
└───────────────────────────────────────────────────────────────┘