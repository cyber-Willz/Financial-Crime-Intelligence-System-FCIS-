# cdd_sanctions

Customer Due Diligence tracking and sanctions-list screening, as a Rust library plus a CLI (`cddctl`).
Rust 1.75 compatible (matches Ubuntu 24.04's toolchain); no async, no database required. *(v4 added the `ml` module; it was built and tested on rustc 1.97 and uses no new dependencies, but could not be compiled on 1.75 here, so confirm with `cargo +1.75 test` before relying on that claim.)*
Depends on the sibling crate `spectral_hypergraph` (`path = "../spectral_hypergraph"`): keep both directories side by side.

## What it does

**Sanctions screening** (`screening`, `lists`)
- Parsers: OFAC SDN/Consolidated CSV (`sdn.csv`+`alt.csv`+`add.csv`), UN Security Council consolidated XML,
  EU FSF XML, and a canonical JSONL format for any other government/territory list (UK, Canada, Australia, local lists…).
- One canonical `ListEntry` (names + aliases with strength, partial DOBs, nationalities, identifiers, programs) with a
  content hash for change detection.
- Index with blocking (exact token, consonant-skeleton, 1-edit deletion neighbourhood) then multi-signal scoring:
  order-insensitive weighted token alignment, spacing/hyphen variants, Cyrillic transliteration, transliteration-variant
  phonetics (Mohammed/Muhammad), initials, legal-form stripping for entities, weak-alias discount, DOB
  (exact / year-only / near-miss / conflict), nationality, identifiers (passport, national ID, IMO, SWIFT, crypto address…).
- Two outcomes only: `Review` and `High`. **The engine never auto-confirms a match**; a human does.
  A bare name match on an individual is never `High`; corroboration (DOB, nationality, identifier) is required.

**CDD tracking** (`cdd`, `engine`)
- Parties (individuals/entities), relationships (owner %, director, controller, trustee, settlor…), documents with
  verification and expiry.
- Beneficial ownership through any number of layers, aggregated over multiple paths, cycle-safe; unresolved corporate owners
  are surfaced rather than ignored.
- OFAC-style **50% rule**: aggregate ownership by blocked parties, no double counting.
- Configurable risk scoring → tier (Low/Medium/High/Prohibited) → due-diligence level (Simplified/Standard/Enhanced),
  with hard floors (PEP, very-high-risk jurisdiction, sanctioned owner) and hard stops (confirmed match, ≥50% blocked ownership).
- Cases with a readiness check that lists *every* blocker (unscreened, open alerts, missing/unverified/expired documents,
  unresolved ownership, missing beneficial-owner ID, party on hold). Four-eyes approval, role-gated:
  Analyst < Reviewer < MLRO; High-risk and confirmed-match decisions need the MLRO.
- Review dates by tier; `reviews_due`.

**Ongoing monitoring**
- `load_list` diffs a new download against the old one (added/changed/removed by content hash), screens **every live party against only the
  delta**, flags open alerts whose entry was delisted, and refuses empty or sharply shrunken lists (truncated download guard).
- A false-positive disposition is valid only while both the listing *and* the customer's screening-relevant details are unchanged;
  otherwise the alert re-opens.
- Owners/controllers are screened with the customer (`screen_party`).

**Audit**: every action is appended to a SHA-256 hash-chained log; `verify()` pinpoints the first altered or deleted event.

## Quick start

```sh
cargo build --release
B=target/release/cddctl; S=state.json
$B --state $S load-list OFAC-SDN ofac sdn.csv alt.csv add.csv --label 2026-10-13
$B --state $S load-list UN-SC un consolidated.xml
$B --state $S screen --name "Ivan Petrov" --dob 1965-03-12 --kind individual
$B --state $S party-add fixtures/party_entity.json && $B --state $S party-screen 1
$B --state $S risk 1 ; $B --state $S alerts --open ; $B --state $S audit-verify
```
Library: see `tests/end_to_end.rs` for the full workflow (onboarding, alerts, confirmation, delta re-screening, persistence).

## Belize FIU Consolidated Sanctions List

Source: https://fiubelize.org/consolidated-belize-sanctions-list/ (XML, XLSX and PDF editions; the FIU's Resources menu also links the OFAC SDN and UN consolidated lists).

```sh
cddctl --state state.json load-list BZ-FIU belize belize_consolidated_sanctions_list.xml --label JULY2026
scripts/update_belize_list.sh state.json target/release/cddctl   # fetch the current XML link from the FIU page and load it (untested against the live site)
cddctl --state state.json stale 35                                # exit 3 if any list has not been refreshed in 35 days
```

What the loader does and what to know:
- The file uses the UN consolidated-list schema with Belize references (`BZCDi.001`); `lists::belize` parses it through the UN parser.
- **Defect in the July 2026 file:** a stray `BZ` is inserted inside some words (`TAI`, `SSI`, `IRE`, `IRI` groups), including names and aliases
  (e.g. `NSANZUBUKBZIRe`, `ByBZIRingiro`). Names are kept as published and a repaired spelling is added as an extra variant, so customers match under either
  spelling. Worth reporting to the FIU. Review the first load's "repaired" count against what you expect.
- Approximate birth years are widened by ±2 years (so a 1-2 year discrepancy is not scored as a DOB conflict); free-text dates such as "Nov. 1973" are parsed.
- **Designated vessels** are published only as a PDF, not in the XML. Transcribe them using `fixtures/belize_vessels.TEMPLATE.jsonl`.
- The Belize list largely mirrors UN designations, so loading both `BZ-FIU` and `UN-SC` gives two alerts for the same person (one per list); each is resolved
  separately. Belize-specific domestic designations (High Court orders) appear only through the FIU file.
- The FIU publishes a new edition after each Gazette order, so refresh often. `ScreeningConfig.max_list_age_days` (default off) blocks CDD approval when a list is older
  than the limit, and approval is always blocked when no list is loaded (`NO_LISTS_LOADED`).
- Under the FIU page's description of the MLTPA, dealing with listed persons is restricted and there are disclosure duties. The engine flags `report_required` on a confirmed match;
  what you must report, to whom and by when is for your MLCO/counsel.

## Spectral hypergraph network analysis (`hypernet`)

The ownership walk is exact for ownership, but it cannot see *association*: a customer that shares a director with a company
owned by a sanctioned person, co-owners in one structure, or two "unrelated" parties quoting the same passport. Those are
many-to-many facts, so the customer base is modelled as a hypergraph with the `spectral_hypergraph` crate:
`Owners(entity)` and `Control(entity)` groups, `Stake` pairs weighted by ownership %, and `SharedIdentifier` groups
(junk values such as `N/A` and groups larger than `max_shared_identifier_group` are ignored).

- **Indirect sanctions exposure**: random walk with restart from sanctioned *seeds* (confirmed matches and blocked parties at 1.0,
  unresolved high alerts at 0.3, review alerts off), solved matrix-free by conjugate gradients on `(1-a)I + a*Delta`.
  The score is the share of walk mass landing on a party (mass is conserved, large groups dilute, an unconnected party scores exactly 0),
  and every score comes with the shortest explanatory path (`cddctl network ID`).
- **Risk rating**: exposure >= `moderate_exposure` adds 15 points and a Medium floor (standard DD); >= `strong_exposure` adds 30 points and a High
  floor (enhanced DD, MLRO approval). Sanctioned parties already in the party's own ownership/control network are excluded (the 50% rule and
  related-confirmed hold cover them), so nothing is double counted. It is **advisory**: it never confirms a match, blocks or holds a party.
- **Structure**: per connected group, algebraic connectivity (lambda2), the sparsest normalised cut (Cheeger sweep over the Fiedler vector) and
  *connector* parties that sit in groups on both sides of it (typical of nominee arrangements). Reported, never scored.

```sh
cddctl --state state.json rel-add 1 2 owner 80      # FROM TO KIND [PCT]
cddctl --state state.json rel-add 3 2 director
cddctl --state state.json network                   # groups + exposed parties
cddctl --state state.json network 4                 # why is party 4 exposed?
cargo run --release --example network_demo          # worked scenario
cargo run --release --example bench_network 20000   # scaling check
```
Library: `Compliance::network_exposure(id)`, `network_analysis()`, `network_seeds()`, `calibrate_network(labels, opts, dampings)`; config under `RiskConfig.network` (`config/risk.example.json`).
Use `network_analysis()` (one pass) rather than calling `assess_risk` per party for bulk re-rating.

**Measured (synthetic, shared directors form one giant group; single thread, release build):**

| parties | largest group | exposure for 1 party (incl. edge build) | exposure, all parties, 7-10 seeds | full `network_analysis` |
|---|---|---|---|---|
| 5,000 | 3,342 | 13 ms | 15 ms | 0.25 s |
| 20,000 | 13,288 | 100 ms | 103 ms | 1.6 s |
| 100,000 | 66,642 | 617 ms | 715 ms | 9.9 s (8 s is the structure analysis; skipped above `structure_max_size` = 25,000) |

With `--links` (`cargo run --release --example bench_network 100000 --links`: two unique links per party, ~4% sharing a bank account, 10% of entities on busy agents), measured on a different machine
from the table above, so compare only the with/without figures below: edge construction 69 ms -> 463 ms at 100,000 parties, full `network_analysis` 2.4 s -> 2.8 s (largest group skips structure analysis).
`exposure_for` rebuilds the edges on every call, so rating one party at that size costs about 0.4 s more with links; use `network_analysis()` for bulk re-rating.

### More ways two customers can be associated (`links`)

Beyond ownership, directors and shared ID numbers, a party can carry **links**: an address, phone, email, bank account or
registered agent. Parties quoting the same value become associated in the network analysis, so a customer who shares a settlement
account or phone number with a sanctioned party gets indirect exposure (and an explanatory path such as
`Ivan Petrov --[shared bank_account (…111)]--> Customer`).

```sh
cddctl --state state.json link-add 7 bank_account "0011 0011 11" --country BZ
cddctl --state state.json link-add 7 registered_agent "Island Corporate Services Ltd" --country BZ
cddctl --state state.json link-add 9 phone "622-1234"      # kinds: address phone email bank_account registered_agent
```
Party JSON takes `"links": [{"kind": "phone", "value": "+501 622 1234", "country": null}]`; old party JSON without `links` still loads.

- **Links are not identifiers.** They never take part in screening and never change the screening profile hash, so editing one cannot re-open a
  false-positive disposition. (A phone number stored as an `Other` identifier *would* match a list entry's `Other` identifier; that is why they are separate.)
- **Normalisation** makes different spellings of one value agree: `12 Main St.` = `No. 12 MAIN STREET`; `P.O. Box 9` = `PO Box 9`; `+501 622-1234` = `00501 6221234`;
  `Ivan.P+kyc@Gmail.com` = `ivanp@gmail.com`; bank accounts ignore spacing; agent names drop `Ltd`/`Limited`/`LLC`… Placeholders (`N/A`, `none`, `0000000`,
  a one-word address) are not links. A local phone number matches its international form only if you set `network.home_dial_code` (e.g. `"501"`).
- **Weight and cap per kind** (`network.links`): bank account 1.0 / 6 parties, phone 0.6 / 6, email 0.6 / 6, address 0.5 / 5, registered agent 0.3 / 15. A value quoted by more
  parties than the cap is skipped (a busy agent, a shared office or a bad data feed is not an association) and counted in the "skipped groups" note. Weight `0` turns a kind off.
  **These weights and caps are judgement-based starting points, not validated.** Use `cddctl calibrate` (below) to see what each kind contributes on your data.
- Analyst-facing hints are masked: numbers show their last 3 characters, emails their domain; addresses and agent names are shown in full (the analyst must be able to read them).
- Known limits: P.O. boxes are weak links in small countries (many unrelated customers share one) and are treated like any other address; phone matching is exact after
  formatting (no fuzzy or last-N-digits matching); addresses are matched by normalised text, not geocoded, so `Unit 4` and `Suite 4` are different; family members legitimately share a home address
  and will be linked.

### Calibrating the thresholds (`cddctl calibrate`)

The damping, thresholds and link weights are starting points. Give the engine parties your MLCO has already judged and it measures them:

```sh
echo '[{"party": 12, "linked": true}, {"party": 40, "linked": false}]' > labels.json
cddctl --state state.json calibrate labels.json [--recall 0.9] [--precision 0.8] [--dampings 0.3,0.6,0.9]
```
It reports the **AUC** of the exposure score (0.5 = chance, 1.0 = perfect), the best-F1 point, suggested `moderate_exposure` / `strong_exposure`, an AUC sweep over damping, and
for each edge family present in your data the AUC *without* it (negative change = it helps, ~0 = adds nothing, positive = it hurts). Exposure is computed as risk rating does it
(sanctioned parties in a party's own ownership/control network are excluded), and labelled parties that are themselves seeds are skipped.

- **Advisory only.** Nothing is applied; read the figures with your MLCO and edit the config yourself.
- `moderate_exposure` is recall-oriented: the weakest labelled link needed to reach the recall target, moved down into the gap to the next lower observed score. `strong_exposure` is
  precision-oriented: the lowest observed score with precision at or above the target and at least 5 true positives. They can contradict each other on weak data; a warning says so.
- **Needs enough labels.** Below 30 of either class a warning is raised; below 10 no thresholds are suggested. Use parties not already in the seed set, judged *before* any
  confirmation that the network itself might have caused, otherwise you measure the network against its own output.
- AUC and thresholds are computed on the same labels (no hold-out split or confidence interval), so they are optimistic. With a lot of labels, calibrate on one half and check on the other.

**Be precise about what is and is not established**
- The solver is checked against an independent dense matrix inverse, walk mass sums to 1, the one-solve and per-seed forms agree, and dense and Lanczos eigenvalues agree (unit tests).
- **The thresholds (damping 0.6, moderate 0.001, strong 0.01, seed strengths, edge and link weights) are starting points set on a synthetic network, not validated against real
  cases.** Exposure depends on how exclusive a tie is (a director with many other boards dilutes it). Run `cddctl network` on your real data, look at the
  distribution, then run `cddctl calibrate` on labelled cases and tune `RiskConfig.network` with your MLCO before relying on the tier floors. Set `network.enabled = false` to switch the risk factor off.
- **Screening fix found on real OFAC data:** a 2-letter token such as `RF` (from `VEB.RF`) used to count as a phonetic match for any word with the same consonants (`Reef`), which let "Reef Developments Ltd" score 0.88 against an unrelated state bank's alias. Phonetic credit now needs at least 3 characters on both sides. Regression test: `short_abbreviations_do_not_phonetically_match_words`. Name-only matches on other generic shared words are still capped at 0.80, as before.
- Link quality depends on data quality: relationships, identifiers and links must be recorded, and normalisation only catches spellings of the same value, not different values for the same place.
- Ownership percentages influence only the 2-party `Stake` edges; the group hyperedges treat all members equally (the crate has uniform incidence weights).

## Learned alert prioritisation (`ml`)

Screening raises alerts with a hand-set score. This layer adds a **learned match-likelihood model** so the review queue can be
read most-likely-true first, trained on what your analysts actually decided.

**It orders alerts. It decides nothing.** It never creates, suppresses, dismisses, confirms, up- or down-levels an alert, and every
alert still needs a human disposition under the same role rules (High / confirm = MLRO). It cannot find a match the rules did not raise:
only alerts that were raised are ever labelled, so it reduces review *effort*, not missed-match risk. Alerts that are High-level or carry a
strong identifier match are always in the `Critical` band whatever the model says.

How it works
- **Features** (`MatchFeatures`, 21 interpretable inputs) are captured on every hit and stored on the alert: rule score, name score, strong
  token matches, token counts, weak/own alias, exact name, DOB (exact / year / near-miss / conflict), nationality, country, identifier
  (strong / weak), entity vs individual, **name rarity** (IDF of the shared tokens over the list: a shared rare name is far more specific than a
  shared common surname) and **competing hits** (a very common name hits many entries). Training sees exactly what the analyst saw.
- **Model**: logistic regression. Every priority is explainable as per-feature contributions (`cddctl ml-explain ALERT`).
- **Training** is a MAP fit with the hand-set scorecard (`prior-v1`) as the prior: coefficients are pulled toward the scorecard with a ridge strength
  chosen by cross-validation, so with few labels it stays near the scorecard and with many it follows your data. Safety-critical coefficients keep their
  sign (a strong identifier match can never lower priority, a conflicting DOB can never raise it); a violating coefficient is frozen at zero and refitted.
- **Validation**: nested k-fold cross-validation **grouped by customer** with confirmed matches spread over the folds. Reports AUC, average precision, Brier,
  reliability bins, and the share of the queue an analyst must read to reach 95% / 100% of true matches, for the model, the scorecard and the current rule score,
  plus a 95% bootstrap interval of the AUC gain over the rule score.
- **Activation gates** (all must pass, `ActivationGates`, defaults: 200 examples, 20 confirmed, 20 false positives, AUC-gain interval strictly above 0, no
  workload regression to find every true match). `ml-train --activate` installs a model only if every gate passes; otherwise nothing changes. Training and activation are
  MLRO-only and audited (with a coefficient fingerprint).
- **Modes**: `shadow` (default) stores the priority at alert time but orders the queue by rule score; `active` orders by the validated model (refused unless a trained
  model passed the gates); `off` stores nothing. **Shadow mode is the prospective test**: `ml-shadow` compares the priority stored *at alert time* with the rule score on alerts resolved
  since, which the model never saw.
- **Low band** (off by default): review-level alerts with no identifier match and no exact DOB, below a cut-off, are labelled `Low` and read last. **They stay open and still need a human decision.**
  `ml-train` suggests a cut-off (half the lowest out-of-fold score of any confirmed match) only when the gates passed. Sample the Low band for QA.

```sh
cddctl --state state.json alerts --queue                 # unresolved alerts in reading order with band and priority
cddctl --state state.json ml-status
cddctl --state state.json --role mlro ml-train                     # dry run: report only
cddctl --state state.json --role mlro ml-train --history old_decisions.json --activate
cddctl --state state.json --role mlro ml-mode active --cutoff 0.002
cddctl --state state.json ml-shadow                      # prospective check on alerts resolved since scoring
cddctl --state state.json ml-explain 12
cddctl --state state.json ml-export labels.jsonl         # features + label, for offline analysis
cargo run --release --example ml_demo 8000 11            # synthetic end-to-end run
```
`--history` takes past decisions from another tool: `[{"subject": {...Subject...}, "list": "OFAC-SDN", "uid": "123", "confirmed": false}]`. Features are recomputed against the
**currently loaded** lists, so records whose entry was delisted, or that would not raise an alert today, are skipped and counted.
Library: `ml::train`, `Compliance::{train_match_model, alert_queue, set_ml_mode, shadow_report, explain_alert, import_history}`.
Old state files load unchanged (alerts raised earlier have no features and are simply not usable as labels).

**What the synthetic check shows (`examples/ml_demo.rs`, 8,000 customers, 5,000 listed people, 3 seeds): machinery works, not real-world gain.**
True matches are perturbed copies of list entries; everything else is drawn from the same name pools. About 0.75 alerts per customer (higher than most real portfolios).
Out-of-fold AUC: learned 0.995-0.999, hand-set scorecard 0.98-0.99, current rule score 0.85-0.86; on a later hold-out period 0.996-0.999 / 0.986-0.991 / 0.84-0.88. Queue read to reach 95% of
true matches: 3-5% (learned) vs 6-15% (scorecard) vs 27-46% (rule score), out-of-fold. The "100% of true matches" figure swings a lot between seeds (5% to 41%) because it
depends on the single hardest match. **Read this carefully:** most of the gain over the rule score comes from the *new features* (name rarity, competing hits, DOB handling), which the untrained scorecard
already uses; training adds a smaller increment. The generator rewards exactly those features, so real gains will be smaller and depend on how your alerts really look. A first run in an
earlier version of the generator, with tiny name pools (4.3 alerts per customer), looked even better and was discarded as unrealistic.

**Be precise about what is and is not established**
- Established by tests (114 total; 77 unit and 37 end-to-end, including 9 new end-to-end for ML): gradient of the penalised likelihood vanishes at the Newton solution; a strong prior keeps the scorecard
  and a weak one follows the data; wrong-sign coefficients are frozen; folds keep customers whole and spread positives; ranking metrics against hand-computed values (ties resolved pessimistically);
  noise labels do **not** pass the gates and a real signal does; the model changes no alert status or level; Critical alerts are never demoted; only an MLRO can train or change mode; Active is refused without a validated
  model; the audit chain verifies; state round-trips; pre-feature state files load.
- **Not established:** performance on real alerts. The scorecard coefficients are hand-set expert guesses. The default gates are conventions, not regulatory requirements; 20 confirmed matches is a low bar for a stable estimate and
  real portfolios often have fewer than that (use `--history`, or stay in shadow mode and let evidence accumulate). Confidence intervals come from out-of-fold predictions of one dataset.
- **Labels are analyst decisions**, so the model inherits analyst habits and errors; reading in priority order keeps labels unbiased only while *every* alert is eventually reviewed.
- Retrain and re-validate when lists, your customer mix or the screening thresholds change (`ml-shadow` is the drift check). Document the training report for your regulator / model-risk owner.

## Verification status (be precise about this)

| Area | Evidence |
|---|---|
| Engine, scoring, ownership, workflows, audit, persistence | 114 automated tests (77 unit, 37 end-to-end, incl. 19 for the spectral network layer, 21 for links and calibration and 9 for the learned prioritiser). The original 91 were reported passing on rustc 1.75; the full 114 were run on rustc 1.97 only |
| Learned alert prioritiser | unit and end-to-end tests above, plus a synthetic-world run (`examples/ml_demo.rs`). **Not validated on real alerts** |
| Scale | 100k synthetic entries: index ~2.8 s, ~80 µs per screen, expected entry found 500/500 (`examples/bench.rs`; synthetic names, so real-world candidate counts will be higher for very common names) |
| `sql/001_cdd_sanctions.sql` | applied to a live PostgreSQL 16 and every constraint accepted/rejected the intended rows (self-relation, owner without %, >100%, self-approval, closed alert without rationale, duplicate open alert, audit update/delete blocked) |
| `sql/002_party_links.sql` | applied on top of 001 to a live PostgreSQL 16 (Ubuntu 24.04 package) and re-run safely; the `links` check accepted 3 valid rows and rejected 5 invalid ones (unknown kind, blank value, missing value, not an array, one bad element among good ones) |
| Links and calibration | normalisation per kind, engine integration (every kind links a customer to a sanctioned party; exposure ordered by weight; large agent groups skipped; links never reach screening or the profile hash; old data and partial configs load), calibration on a network with a known answer (AUC 1.0; removing the bank-account family drops it to 0.0), `link-add` and `calibrate` exercised through the real CLI. **Not validated: the default link weights and caps, and the thresholds suggested by `calibrate` on real data** |
| OFAC and UN parsers | tested on **synthetic fixtures** written from the published formats; not run against a live download (no access to those hosts from the build sandbox) |
| Belize FIU parser | structure and defects taken from the real July 2026 XML (first ~50 individuals inspected via the FIU site); entities and later sections not inspected; not run on the full file |
| EU parser | synthetic fixture only; schema recalled from documentation. Validate on a real file first |

## Before you rely on it (the "perfect system" caveats)

1. **Run the parsers on the real downloads** and compare entry counts with the publisher's totals. Download URLs/formats change; fetch
   and schedule them in your own job (the library takes the file contents, it does not fetch).
2. **Tune and measure thresholds** (`ScreeningConfig`: review 0.85, high 0.95) on your own data. Build a labelled test set of true
   matches and known look-alikes; screening quality is an empirical property. Document the tuning for your regulator.
3. **Which lists** is a legal question for your jurisdiction and correspondent banks, not a code one. The engine loads any list you give it.
4. **Country risk is not hard-coded.** Supply `RiskConfig.country_risk` (see `config/risk.example.json`; the codes there are placeholders)
   from your regulator's and FATF's current lists. Unlisted countries default to risk 0.
5. **Not covered:** (the learned prioritiser orders alerts; it does not reduce missed-match risk) Arabic/CJK transliteration (non-Latin names match only through Latin aliases, except Cyrillic), adverse-media
   search, PEP-list data (PEP status is an input flag), transaction monitoring, STR/SAR filing (the engine flags `report_required`; it files nothing),
   identity verification/biometrics, authentication/authorisation (roles are passed in by the caller), and encryption at rest.
6. **Legal review:** thresholds (25% UBO, 50% rule), review cycles and document sets are configurable defaults, not legal advice.
7. Persistence is a JSON snapshot (atomic write). For multi-user production use, store state relationally (schema provided) behind
   your own service and keep the engine as the decision layer.
