# aml_bo — beneficial-ownership analysis (v0.5)

Connects the vendored `cdd_sanctions` engine (parties, ownership/control relationships, sanctions screening,
four-eyes alert workflow, spectral network exposure) to transaction monitoring.

```
CDD register (parties, owners, alerts) ─┐
account → party registry (accounts.csv) ─┴─► OwnershipAnalysis ─► per-holder: UBOs, opacity, 50% rule, exposure, group
                                                    │
                      ┌─────────────────────────────┼───────────────────────────┐
                      ▼                             ▼                           ▼
              BoMonitor (rules)        graph_map() → aml_graph features   analyst reports (`aml-bo analyze`)
```

## What it adds over the CDD engine
* **Indexed ownership walk** per account holder (the engine's walk rescans all relationships at each step). Effective
  percentages, unresolved fraction and layer count are tested to equal `Compliance::ownership_report`. Extra: path counts,
  and circular holdings are *reported* (the engine skips them silently).
* **Opacity score** (0–1) with reasons: layers, share ending at corporate owners with no recorded owners, no beneficial
  owner identified, nominee/bearer shares, high-risk jurisdiction in the chain, circular holdings, stakes split just below the UBO threshold.
  Weights are judgement-based defaults (`BoConfig`), not validated on real cases.
* **50% rule**: share of a holder owned by blocked parties (confirmed matches in the engine's alert workflow), through chains.
* **Ownership groups**: union-find over ownership ≥10% and control links; oversized groups are dropped and counted.
* **`BoMonitor`**: causal streaming rules (blocked ownership, opaque owner, fragmented ownership, pass-through on opaque/grouped accounts,
  circulation inside a group, owners paying just under 10k into their own company, network exposure), combined by noisy-OR. Untuned thresholds.
* **Graph-model features** (`aml_graph`, `NODE_FEATS` 20→32, `EDGE_FEATS` 6→10): ownership attributes per node; same-party / shared-owner /
  same-group / sanctioned-stake flags per edge; one hyperedge per ownership group in the temporal hypergraph.

## CLI
```
aml-bo synth   --out DIR                       # transactions.csv (SAML-D layout), compliance.json, accounts.csv
aml-bo analyze --compliance DIR/compliance.json --accounts DIR/accounts.csv [--party ID]
aml-bo eval    [--seeds 3] [--json-out F]      # engine / BO rules / graph / graph+ownership / +BO / placebo
```
`accounts.csv` needs `account,party` columns (one holder per account; joint accounts are out of scope).

## Results (synthetic world, 3 model seeds; `eval_synthetic_3seeds.{txt,json}`)
Test = last 39 daily windows, 27,493 txns, 589 laundering. Typologies 8–11 (shell groups, sanctioned-owner evasion, nominee pass-through,
split stakes) are planted by the same generator that writes the ownership register, next to benign look-alikes (corporate groups,
deep chains with known owners, quiet nominee companies, 30–45% sanctioned stakes, namesake false positives, 4×24% families).

| | AP | ROC-AUC |
|---|---|---|
| rule engine, online | 1.8% | 0.50 |
| BO rules only | 17.0% | 0.62 |
| engine + BO rules | 7.0% | 0.61 |
| graph model, no ownership | 86.6 ± 2.6% | 0.986 |
| graph model + ownership | 85.3 ± 4.0% | 0.984 |
| graph model + ownership + BO rules | 85.5 ± 3.9% | 0.985 |
| placebo (ownership shuffled over accounts) | 84.7 ± 0.7% | 0.983 |

**Read this honestly.**
* BO rules help the rule engine a great deal here (sanctioned-owner evasion 23%→100% recall, split stakes 0%→77%), because the engine
  has no ownership knowledge. That is expected on a world built to contain such cases.
* They add **nothing measurable to the graph model**: differences are inside seed noise. In this world the flows of typologies 8–11 are
  already learnable from the transaction graph (~100% recall at the engine's alert volume with or without ownership). Per-typology
  differences on the older typologies (Layered_Chain, Slow_Drip) are small and not significant with 3 seeds.
* Real value would show where flows look ordinary but ownership is not (e.g. a clean-looking company 55% owned by a listed person).
  The generator has few such cases and no way to prove it either way. Real KYC data is needed.
* Ownership is a static snapshot treated as known from day one; real registers are incomplete and lag.
* The network-exposure feature never fired (`exposed: 0`): the synthetic register has no shared contact links, so that part is untested end to end.
* Single data seed (11); model seeds vary only initialisation and sampling.

## Tests
`cargo test --release -p aml_bo`: walk equals the engine's report; circular holdings; 50% rule and near misses; groups and fragmented
ownership; monitor recall/false-positive rate and causality (prefix scoring equals full-stream scoring); determinism and CSV/JSON roundtrips;
ownership changes graph-model inputs.
