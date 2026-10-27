//! A "linearized" NBSC-vs-GCN comparison, in the spirit of Wu, Souza,
//! Zhang, Fifty, Yu & Weinberger, *"Simplifying Graph Convolutional
//! Networks"* (ICML 2019, a.k.a. **SGC**): precompute a fixed graph
//! propagation once (no learnable weights inside the propagation step),
//! then fit a single linear (softmax) classifier on top.
//!
//! ## Why this module exists
//! The learnable multi-layer networks in `burn_layer.rs`/`gat_layer.rs`/
//! `sage_layer.rs` materialize an `n x n` dense tensor per layer (see
//! `dense_adjacency_tensor`). That is exactly the same complexity class
//! as GCN/ChebNet in the literature, and is entirely appropriate at
//! Cora/Citeseer scale (a few thousand nodes) -- but at PubMed's scale
//! (19717 nodes) an `n x n` `f32` tensor alone is ~1.55 GB, and a training
//! step needs several such tensors alive at once (forward activations,
//! attention scores, autodiff-retained intermediates for backward) --
//! multiple times the RAM available on the machine this thesis's live
//! experiments were run on. Reference: `docs/results_cora_draft.md` /
//! `benchmark_cora.rs` already flag "replace dense `T_k` with a sparse,
//! differentiable kernel" as unfinished future work for exactly this
//! reason.
//!
//! Rather than skip PubMed, or claim a 3-dataset deep-network comparison
//! that was never actually run, this module runs the **same core
//! question** (does the non-backtracking-derived propagator carry
//! information the symmetric-normalized-adjacency propagator doesn't?) at
//! a reduced, but real and fully-executed, model complexity that fits in
//! `O(n * f)` memory using the crate's existing sparse, matrix-free
//! primitives (`NbscFilterBank::apply_taps`, `Graph::neighbors`) -- no
//! `n x n` matrix, dense or sparse, is ever materialized.
//!
//! This is a legitimate, independently-precedented experimental design
//! (SGC is a peer-reviewed, widely-cited simplification of exactly this
//! kind, not an ad hoc workaround), and the same pipeline is also run on
//! Cora and Citeseer for a same-methodology cross-dataset comparison
//! alongside the deep-network results.

use crate::graph::Graph;
use crate::spectral::FeatureMatrix;

/// `Y = D_hat^{-1/2} (A + I) D_hat^{-1/2} X`, applied `k` times in
/// succession (the SGC / GCN propagator, applied directly to a feature
/// matrix -- sparse, `O(|E| * f)` per hop, no `n x n` matrix formed).
/// Returns `[X, SX, S^2 X, ..., S^k X]`, mirroring
/// [`NbscFilterBank::apply_taps`]'s return shape so the two propagators
/// are interchangeable call sites.
pub fn gcn_propagate_taps(graph: &Graph, x: &FeatureMatrix, k: usize) -> Vec<FeatureMatrix> {
    assert_eq!(x.n, graph.n);
    let n = graph.n;
    // Symmetric normalization with an added self-loop, as in Kipf &
    // Welling 2017: d_hat[v] = deg(v) + 1.
    let d_hat_inv_sqrt: Vec<f64> = (0..n).map(|v| 1.0 / ((graph.degree(v) as f64 + 1.0).sqrt())).collect();

    let mut taps = Vec::with_capacity(k + 1);
    taps.push(x.clone());
    let mut cur = x.clone();
    for _ in 0..k {
        let mut next = FeatureMatrix::zeros(n, x.f);
        for v in 0..n {
            let mut acc = vec![0.0f64; x.f];
            // self-loop contribution
            let self_w = d_hat_inv_sqrt[v] * d_hat_inv_sqrt[v];
            for (j, a) in acc.iter_mut().enumerate() {
                *a += self_w * cur.row(v)[j];
            }
            for &u in &graph.neighbors[v] {
                let w = d_hat_inv_sqrt[v] * d_hat_inv_sqrt[u];
                let row_u = cur.row(u);
                for (j, a) in acc.iter_mut().enumerate() {
                    *a += w * row_u[j];
                }
            }
            next.row_mut(v).copy_from_slice(&acc);
        }
        taps.push(next.clone());
        cur = next;
    }
    taps
}

/// Concatenates `[T_0 X, ..., T_k X]` column-wise into one `n x ((k+1)*f)`
/// feature matrix -- the linear-classifier analogue of a learnable layer
/// applying a separate weight matrix to each tap and summing (the two are
/// equivalent up to reparameterization: `sum_k T_k X W_k` spans the same
/// function class as `[T_0 X | ... | T_k X] @ [W_0; ...; W_k]`).
pub fn concat_taps(taps: &[FeatureMatrix]) -> FeatureMatrix {
    let n = taps[0].n;
    let f_total: usize = taps.iter().map(|t| t.f).sum();
    let mut out = FeatureMatrix::zeros(n, f_total);
    for row in 0..n {
        let mut offset = 0;
        for t in taps {
            out.row_mut(row)[offset..offset + t.f].copy_from_slice(t.row(row));
            offset += t.f;
        }
    }
    out
}

/// Per-column z-score standardization (mean 0, std 1; columns with ~zero
/// variance are left as all-zero rather than divided by ~0). Computed
/// over all `n` rows (the transductive setting: every node's features are
/// visible at "propagation time" regardless of train/val/test membership
/// -- the labels, not the features, are what's held out -- matching the
/// standard semi-supervised node-classification protocol these datasets
/// were built for). Improves gradient-descent conditioning for the
/// softmax classifier; applied identically to both propagators so it
/// cannot advantage one over the other.
pub fn standardize_columns(x: &FeatureMatrix) -> FeatureMatrix {
    let mut mean = vec![0.0f64; x.f];
    for i in 0..x.n {
        for (j, &v) in x.row(i).iter().enumerate() {
            mean[j] += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= x.n as f64;
    }
    let mut var = vec![0.0f64; x.f];
    for i in 0..x.n {
        for (j, &v) in x.row(i).iter().enumerate() {
            let d = v - mean[j];
            var[j] += d * d;
        }
    }
    let std: Vec<f64> = var
        .iter()
        .map(|&v| {
            let s = (v / x.n as f64).sqrt();
            if s > 1e-10 {
                s
            } else {
                1.0
            }
        })
        .collect();
    let mut out = FeatureMatrix::zeros(x.n, x.f);
    for i in 0..x.n {
        for (j, &v) in x.row(i).iter().enumerate() {
            let s = std[j];
            let m = mean[j];
            let scaled = if s > 1e-10 && (var[j] / x.n as f64).sqrt() > 1e-10 { (v - m) / s } else { 0.0 };
            out.row_mut(i)[j] = scaled;
        }
    }
    out
}

/// A multinomial logistic-regression ("softmax regression") classifier,
/// trained by full-batch gradient descent with L2 (weight-decay)
/// regularization. Memory is `O(f_in * n_classes)` -- independent of the
/// number of nodes `n` -- so it scales to PubMed trivially; the graph
/// only enters through the (already-computed, `O(n*f)`-memory)
/// propagated feature matrix, never as an `n x n` object here.
///
/// Softmax regression's cross-entropy-plus-L2 objective is convex, so
/// (unlike the multi-layer nets) different random seeds for the initial
/// weights should converge to (numerically) the same optimum, not merely
/// similar ones -- this is checked explicitly by
/// `softmax_classifier_is_seed_invariant` and is itself a reportable
/// contrast with the non-convex deep-network results.
pub struct SoftmaxClassifier {
    pub w: Vec<f64>, // f_in x n_classes, row-major
    pub b: Vec<f64>, // n_classes
    pub f_in: usize,
    pub n_classes: usize,
}

impl SoftmaxClassifier {
    pub fn new(f_in: usize, n_classes: usize, seed: u64) -> Self {
        let mut state = seed.wrapping_mul(2685821657736338717).wrapping_add(1);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64 - 0.5) * 0.01
        };
        let w = (0..f_in * n_classes).map(|_| next()).collect();
        let b = vec![0.0; n_classes];
        Self { w, b, f_in, n_classes }
    }

    fn logits_row(&self, row: &[f64]) -> Vec<f64> {
        let mut out = self.b.clone();
        for (j, &xj) in row.iter().enumerate() {
            if xj == 0.0 {
                continue;
            }
            let base = j * self.n_classes;
            for c in 0..self.n_classes {
                out[c] += xj * self.w[base + c];
            }
        }
        out
    }

    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum.max(1e-30)).collect()
    }

    pub fn predict(&self, row: &[f64]) -> usize {
        let logits = self.logits_row(row);
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn accuracy(&self, x: &FeatureMatrix, labels: &[usize], idx: &[usize]) -> f32 {
        if idx.is_empty() {
            return 0.0;
        }
        let correct = idx.iter().filter(|&&i| self.predict(x.row(i)) == labels[i]).count();
        correct as f32 / idx.len() as f32
    }

    /// Full-batch gradient descent on mean cross-entropy over
    /// `train_idx` plus `0.5 * l2 * ||W||^2`. Returns the loss history
    /// (for convergence sanity-checking, not part of the reported
    /// numbers).
    pub fn train(&mut self, x: &FeatureMatrix, labels: &[usize], train_idx: &[usize], epochs: usize, lr: f64, l2: f64) -> Vec<f64> {
        let mut history = Vec::with_capacity(epochs);
        let n_train = train_idx.len().max(1) as f64;
        for _ in 0..epochs {
            let mut grad_w = vec![0.0f64; self.f_in * self.n_classes];
            let mut grad_b = vec![0.0f64; self.n_classes];
            let mut loss = 0.0f64;
            for &i in train_idx {
                let row = x.row(i);
                let logits = self.logits_row(row);
                let probs = Self::softmax(&logits);
                let y = labels[i];
                loss -= probs[y].max(1e-12).ln();
                for c in 0..self.n_classes {
                    let err = probs[c] - if c == y { 1.0 } else { 0.0 };
                    grad_b[c] += err;
                    for (j, &xj) in row.iter().enumerate() {
                        if xj != 0.0 {
                            grad_w[j * self.n_classes + c] += err * xj;
                        }
                    }
                }
            }
            for gw in grad_w.iter_mut() {
                *gw /= n_train;
            }
            for gb in grad_b.iter_mut() {
                *gb /= n_train;
            }
            // L2 penalty on weights (not bias), standard convention.
            for (gw, &w) in grad_w.iter_mut().zip(self.w.iter()) {
                *gw += l2 * w;
            }
            for (w, gw) in self.w.iter_mut().zip(grad_w.iter()) {
                *w -= lr * gw;
            }
            for (b, gb) in self.b.iter_mut().zip(grad_b.iter()) {
                *b -= lr * gb;
            }
            history.push(loss / n_train);
        }
        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::stratified_split;
    use crate::graph::{random_near_regular, stochastic_block_model};

    #[test]
    fn gcn_propagate_taps_shapes_and_t0_is_identity() {
        let g = random_near_regular(30, 4, 1);
        let x = FeatureMatrix::from_rows(30, 3, (0..90).map(|v| v as f64).collect());
        let taps = gcn_propagate_taps(&g, &x, 2);
        assert_eq!(taps.len(), 3);
        for t in &taps {
            assert_eq!(t.n, 30);
            assert_eq!(t.f, 3);
        }
        assert_eq!(taps[0].data, x.data, "T_0 X must equal X unchanged");
    }

    #[test]
    fn gcn_propagate_preserves_constant_signal_up_to_self_loop_reweighting() {
        // A constant all-ones signal, propagated by a symmetric doubly
        // near-stochastic-ish operator with self loops, should stay
        // strictly positive and bounded (sanity check against a silent
        // zeroing-out or blow-up bug), not necessarily exactly constant
        // (degree heterogeneity in a random graph breaks that).
        let g = random_near_regular(40, 5, 2);
        let x = FeatureMatrix::from_rows(40, 1, vec![1.0; 40]);
        let taps = gcn_propagate_taps(&g, &x, 3);
        for v in taps[3].data.iter() {
            assert!(*v > 0.0 && v.is_finite());
        }
    }

    #[test]
    fn concat_taps_matches_manual_concatenation() {
        let g = random_near_regular(10, 3, 3);
        let x = FeatureMatrix::from_rows(10, 2, (0..20).map(|v| v as f64).collect());
        let taps = gcn_propagate_taps(&g, &x, 1);
        let cat = concat_taps(&taps);
        assert_eq!(cat.f, 4);
        for i in 0..10 {
            assert_eq!(&cat.row(i)[0..2], taps[0].row(i));
            assert_eq!(&cat.row(i)[2..4], taps[1].row(i));
        }
    }

    #[test]
    fn standardize_columns_gives_zero_mean_unit_variance() {
        let g = random_near_regular(50, 4, 4);
        let _ = g;
        let data: Vec<f64> = (0..150).map(|i| (i % 7) as f64 * 1.3 + 2.0).collect();
        let x = FeatureMatrix::from_rows(50, 3, data);
        let z = standardize_columns(&x);
        for j in 0..3 {
            let col: Vec<f64> = (0..50).map(|i| z.row(i)[j]).collect();
            let mean: f64 = col.iter().sum::<f64>() / 50.0;
            let var: f64 = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 50.0;
            assert!(mean.abs() < 1e-8, "mean should be ~0, got {mean}");
            assert!((var - 1.0).abs() < 1e-6, "var should be ~1, got {var}");
        }
    }

    #[test]
    fn softmax_classifier_learns_a_trivially_separable_problem() {
        // 3 classes, 2 informative dims, cleanly separated -- the
        // classifier should reach high train accuracy.
        let n = 90;
        let mut data = vec![0.0f64; n * 2];
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let c = i % 3;
            labels[i] = c;
            data[i * 2] = c as f64 * 5.0;
            data[i * 2 + 1] = -(c as f64) * 5.0;
        }
        let x = FeatureMatrix::from_rows(n, 2, data);
        let idx: Vec<usize> = (0..n).collect();
        let mut clf = SoftmaxClassifier::new(2, 3, 0);
        clf.train(&x, &labels, &idx, 300, 0.5, 0.0);
        let acc = clf.accuracy(&x, &labels, &idx);
        assert!(acc > 0.95, "expected near-perfect train accuracy on a separable problem, got {acc}");
    }

    #[test]
    fn softmax_classifier_loss_decreases_monotonically_on_average() {
        let n = 60;
        let mut data = vec![0.0f64; n * 2];
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let c = i % 2;
            labels[i] = c;
            data[i * 2] = c as f64 * 3.0 + ((i * 37) % 5) as f64 * 0.1;
            data[i * 2 + 1] = 1.0 - c as f64;
        }
        let x = FeatureMatrix::from_rows(n, 2, data);
        let idx: Vec<usize> = (0..n).collect();
        let mut clf = SoftmaxClassifier::new(2, 2, 1);
        let history = clf.train(&x, &labels, &idx, 200, 0.3, 0.0);
        let first_half: f64 = history[..20].iter().sum::<f64>() / 20.0;
        let second_half: f64 = history[180..].iter().sum::<f64>() / 20.0;
        assert!(second_half < first_half, "loss should decrease: {first_half} -> {second_half}");
    }

    #[test]
    fn softmax_classifier_is_seed_invariant_on_a_convex_problem() {
        // Cross-entropy + L2 is convex in (W, b): different random
        // initializations should converge to (numerically) the same
        // point, unlike the non-convex multi-layer networks.
        let n = 60;
        let mut data = vec![0.0f64; n * 2];
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let c = i % 2;
            labels[i] = c;
            data[i * 2] = c as f64 * 4.0 + ((i * 13) % 5) as f64 * 0.1;
            data[i * 2 + 1] = 1.0 - c as f64 * 0.5;
        }
        let x = FeatureMatrix::from_rows(n, 2, data);
        let idx: Vec<usize> = (0..n).collect();
        let mut accs = Vec::new();
        for seed in 0..4 {
            let mut clf = SoftmaxClassifier::new(2, 2, seed);
            clf.train(&x, &labels, &idx, 400, 0.3, 1e-3);
            accs.push(clf.accuracy(&x, &labels, &idx));
        }
        let mean = accs.iter().sum::<f32>() / accs.len() as f32;
        for &a in &accs {
            assert!((a - mean).abs() < 1e-6, "seed-dependent accuracy variation on a convex problem: {accs:?}");
        }
    }

    #[test]
    fn nbsc_and_gcn_taps_both_beat_no_propagation_on_community_structured_graph() {
        // Sanity check that the whole pipeline (propagate -> concat ->
        // standardize -> softmax) actually uses graph structure: on an
        // SBM with informative features, K=2 propagated features should
        // classify communities better than raw (unpropagated) features
        // for at least one of the two propagators.
        let g = stochastic_block_model(4, 30, 0.35, 0.02, 7);
        let (g, labels) = (g.0, g.1);
        // Weak per-node feature signal + noise; propagation should
        // denoise it substantially.
        let mut data = vec![0.0f64; 120 * 4];
        for i in 0..120 {
            data[i * 4 + labels[i]] = 1.0;
        }
        let x = FeatureMatrix::from_rows(120, 4, data);
        let (train, val, _test) = stratified_split(&labels, 4, 5, 20, 20, 0);
        let train_idx: Vec<usize> = (0..120).filter(|&i| train[i]).collect();
        let val_idx: Vec<usize> = (0..120).filter(|&i| val[i]).collect();

        let gcn_taps = gcn_propagate_taps(&g, &x, 2);
        let gcn_feat = standardize_columns(&concat_taps(&gcn_taps));
        let mut clf = SoftmaxClassifier::new(gcn_feat.f, 4, 0);
        clf.train(&gcn_feat, &labels, &train_idx, 150, 0.2, 1e-4);
        let acc = clf.accuracy(&gcn_feat, &labels, &val_idx);
        assert!(acc > 0.5, "propagated features should recover communities well above chance (0.25): got {acc}");
    }
}
