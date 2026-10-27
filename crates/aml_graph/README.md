# aml_graph — connectome-informed recurrent graph neural network (v0.4)

```
Transaction Graph ─► Temporal Hypergraph ─► Recurrent Memory ─► Graph Neural Network ─► AML Risk
```

| Stage | What it is here |
|---|---|
| Transaction graph | Per time window (default 1 day): accounts as nodes, distinct directed `sender → receiver` pairs, causal node features (volumes, distinct counterparties, cash / cross-border / high-risk-jurisdiction shares, retention, deviation from the account's own history, idle time, trailing-7-window counterparty counts, optional rule-engine score). |
| Temporal hypergraph | A fan-in is one receiver and *all* its senders at once. Each account with ≥ 3 distinct senders (receivers) in a window gets an in-star (out-star) hyperedge; the sequence of daily snapshots plus the multi-horizon hyperedge statistics is the temporal hypergraph. Stage 1 is a hypergraph + graph convolution `[x ‖ Ĥx ‖ A_in x ‖ A_out x]`. A non-backtracking (Hashimoto) walk mass on the directed flow graph is a feature that is non-zero only after long chains and in directed cycles. |
| Recurrent memory | One persistent state per account of the **connectome RNN** (`vendor/hcrnn`, v0.4 of `human_connectome_rnn`): connectome wiring, conduction delays (rescaled to ≤ 3 steps), Dale's-law E/I units, input only into the E units of the input regions, read-out only from the E units of the output regions, Arnoldi radius cap. Weights are shared by all accounts; state advances once per active window (event-driven), so idle accounts cost nothing. |
| Graph neural network | A second graph/hypergraph layer over the **memory read-outs** of neighbours, then a per-transaction head → probability. |

Training is exact truncated BPTT through all four stages (checked against finite differences for every
parameter group and every ablation switch: `tests/gradcheck.rs`). Features are causal (tested). Splits are by
time. The vendored crates are unmodified apart from removing their binaries/examples.

```sh
cargo run --release -p aml_graph -- wiring                      # topology of the wiring and its null models
cargo run --release -p aml_graph -- eval                        # synthetic world, ablations
cargo run --release -p aml_graph -- eval --csv SAML-D.csv --format saml-d --limit 3000000 \
      --variants full,tabular,no-memory,rewired --seeds 3 --save model.json
cargo run --release -p aml_graph -- score --model model.json --csv new.csv --out risk.csv
```
Library: `GraphScorer::push(txn)` returns every transaction's risk when its window closes (real-time rules stay
in `aml_engine`; this is the end-of-window re-scoring / prioritisation layer). Variants: `full`, `tabular`
(features only), `no-memory`, `no-hypergraph`, `no-graph`, `no-rules`, `rewired`, `random`, `dense`.

## What was measured — and what it does not show

No labelled AML data was available when this was built, so everything below is on the **synthetic world** in
`synth.rs` (150 days, 149k transactions, 7 laundering typologies that unfold over days, 1.5 % positives,
benign look-alikes: merchant fan-in, payroll fan-out, legitimate pass-through processors, recurring payments,
loops; scheme accounts also get ordinary background activity). Test = last 38 days, 3 seeds, mean ± sd.
**The generator is mine, so this shows the machinery can learn multi-day structure, not how it performs on a
bank's data.** Run the same command on SAML-D (`--csv`) before believing any number.

| Test | AP | best F1 | recall @ rule-engine alert volume |
|---|---|---|---|
| rule engine, default weights | 1.2 % | 3.7 % | 6.5 % |
| model: full | 82.0 ± 1.0 | 76.4 ± 3.2 | 97.7 ± 0.9 |
| features only (`tabular`) | 10.9 ± 2.6 | 23.7 ± 2.5 | 39.8 ± 0.4 |
| no memory | 18.4 ± 4.6 | 25.5 ± 4.9 | 68.8 ± 0.6 |
| no hypergraph | 66.1 ± 14.4 | 65.5 ± 10.3 | 98.3 ± 0.8 |
| no graph (hypergraph only) | 78.9 ± 5.4 | 74.0 ± 4.5 | 97.8 ± 1.2 |
| no rule features | 82.5 ± 1.4 | 76.2 ± 2.7 | 98.5 ± 0.4 |
| rewired wiring (degree-preserving null) | 86.2 ± 3.3 | 80.9 ± 2.4 | 98.9 ± 0.5 |
| density-matched random wiring | 86.0 ± 3.2 | 80.2 ± 3.2 | 98.8 ± 0.3 |
| dense wiring | 85.7 ± 2.3 | 80.9 ± 2.1 | 98.4 ± 0.4 |

Honest reading:

1. **The rule baseline is not a fair fight.** Its default weights are fitted to SAML-D; on this world it raises
   alerts on 15 % of transactions (benign structure fires it) and ranks at chance. Do not read the gap as
   "the model beats the engine"; the engine was not recalibrated (`aml-eval --fit-out` does that on real data).
2. **Memory and message passing carry the result here.** Same features without graph or memory: AP 11 %.
   Removing memory alone: 18 %. Removing the hypergraph block: 66 % with high variance; removing only the
   pairwise graph: 79 % (no clear loss). The typologies that unfold over days (slow drip, ramp, layering,
   dormant pass-through) are exactly where the rule engine and the single-window features fail.
3. **The connectome wiring did not help.** Degree-preserving, random and dense wirings scored as well or
   slightly better (86 vs 82 AP; differences are within about 1–2 sd at 3 seeds). That agrees with the
   `human_connectome_rnn` README (the structural prior mattered little once the units were gated/trained).
   The connectome remains the default because it is the requested architecture and gives a stable, Dale-
   constrained, sparse (2,040-synapse) memory, but nothing measured here says it is *better* than a random
   sparse memory. Test on real data and on the real TVB-66 connectome (`--tvb DIR`) before claiming it.
4. **Rule features added nothing** on this world (82.5 without vs 82.0 with).
5. 475 test positives, 325 of them one typology (bipartite mesh); per-typology numbers are noisy, and the
   cycle typology (recall 58 %) is the clear weak spot — a window-local hypergraph sees a cycle only when its
   hops fall in one window.
6. One generator iteration was needed to remove a shortcut (scheme-only accounts); others may remain.

## Limits

* Window-level (e.g. daily) scoring, not per-transaction real-time; ≈ 2.6 KB of memory per tracked account
  (idle accounts are evicted after 90 windows).
* Not wired into `aml_api`/PostgreSQL (no migration, no endpoint): use `GraphScorer` or `aml-gnn score`.
* `rule_retro` uses later transactions and is a baseline only, never a feature.
* Wiring defaults to a synthetic human-like connectome; real TVB-66 needs `scripts/fetch_tvb66.sh` from the
  connectome project (network access) and the data-use terms in `vendor/hcrnn/DATA_USE_TERMS.md`.
