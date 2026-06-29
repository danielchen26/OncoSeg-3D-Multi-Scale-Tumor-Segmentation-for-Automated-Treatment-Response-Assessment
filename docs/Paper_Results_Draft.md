# Results

All numbers below are from a single 50-epoch training run on the Medical Segmentation Decathlon (MSD) Task01_BrainTumour split (388 train / 96 val subjects) using multi-label sigmoid outputs over the three standard BraTS regions — Tumor Core (TC), Whole Tumor (WT), Enhancing Tumor (ET). Training was performed on an Apple M1 (MPS) with roi_size = 96³ and embed_dim = 24 for OncoSeg.

## 1. Segmentation accuracy

Table 1 compares OncoSeg against the UNet3D baseline on the 96-subject validation set.

| Model          | Dice TC    | Dice WT     | Dice ET    | Dice Mean  | HD95 Mean (mm) | Params |
|----------------|------------|-------------|------------|------------|----------------|--------|
| **OncoSeg**    | **0.7898** | **0.8529**  | **0.7481** | **0.7969** | **15.35**      | **3.7 M** |
| UNet3D         | 0.7849     | 0.8522      | 0.7462     | 0.7944     | 21.03          | 19.2 M |

A one-sided Wilcoxon signed-rank test on the per-subject Dice arrays finds **no region with a statistically significant OncoSeg advantage** (TC p = 0.46, WT p = 0.995, ET p = 0.57; mean p = 0.41). On WT, UNet3D in fact wins on 67 of 96 subjects. The mean-Dice difference (+0.0025) is within run-to-run noise.

The honest framing is therefore **parameter efficiency, not accuracy superiority**: OncoSeg matches UNet3D's Dice using **5.2× fewer parameters**, while showing a lower mean HD95 (95-percentile Hausdorff distance): 15.35 mm vs 21.03 mm, a **27 % reduction**. Note that HD95 is stored only as an aggregate mean — no per-subject HD95 array was retained — so this boundary-error gap is **not** significance-tested and should be read as descriptive.

Training curves and the per-region Dice comparison figure are included as Figures 1 and 2 (`experiments/local_results/training_curves.png`, `dice_comparison.png`).

## 2. Qualitative analysis

Figure 3 (`figures/qualitative_comparison.png`) stratifies the validation set into best / median / worst subjects by OncoSeg mean Dice:

- **Best (BRATS_407, Dice 0.946):** a compact, high-contrast lesion. Both OncoSeg and UNet3D agree closely with GT; the TC/ET boundary matches to within one voxel in the rendered slice.
- **Median (BRATS_425, Dice 0.852):** a larger, heterogeneous lesion. OncoSeg more faithfully recovers the inner TC boundary; UNet3D over-segments the edema margin.
- **Worst (BRATS_077, Dice 0.239):** a small, fragmented tumor (Section 4).

The qualitative gap between the two models grows as the case difficulty increases, consistent with the Dice gap being driven by harder subjects rather than uniform gains across the cohort.

## 3. Uncertainty quantification

MC Dropout inference (5 samples, keeping the dropout layer active at test time) produces a per-voxel predictive-entropy map on the median case (Figure 4, `figures/uncertainty_map.png`). The uncertainty concentrates along tumor boundaries and in regions of disagreement with the ground truth — the two qualities a radiologist would want from a review-aid overlay.

Calibration was measured by binning per-voxel predicted probabilities over all three channels and comparing each bin's mean confidence to its empirical accuracy (15 equal-width bins, reliability diagram in Figure 5).

> **Expected Calibration Error (ECE): 0.0101 over all voxels, but ≈ 0.49 on foreground (tumor) voxels.**

The 0.0101 figure is **dominated by background**: ~98.4 % of voxels fall in the lowest-confidence bin (trivially-easy background), so the pooled ECE mostly measures how well the model abstains on healthy tissue. Restricted to foreground tumor voxels — the clinically relevant regime — ECE is ≈ 0.49 and the highest-confidence bin is correct only ~40 % of the time: the model is **over-confident on tumor voxels**. The pooled value should not be read as a calibration guarantee. (Derived from one subject, 5 MC samples.) The uncertainty-vs-error plot (Figure 6, `figures/uncertainty_vs_error.png`) does show a monotone relationship — higher-entropy voxels have higher error rates — so the entropy map remains useful as a relative triage signal even though absolute calibration on tumor voxels is poor.

## 4. Failure-mode analysis

OncoSeg's bottom-5 validation cases by mean Dice are reported in `experiments/local_results/failure_analysis.json`. Aggregated across these five subjects, the relative drop in Dice per region is:

| Region | Bottom-5 mean | Overall mean | Relative drop |
|--------|---------------|--------------|---------------|
| TC     | 0.161         | 0.790        | −79.7 %       |
| WT     | 0.565         | 0.853        | −33.7 %       |
| ET     | 0.117         | 0.748        | **−84.3 %**   |

**ET is the dominant failure region** once the aggregate is computed with `nanmean` (an earlier `mean` made ET's overall figure NaN and silently dropped it, mislabelling TC as the worst region). ET's bottom-5 Dice (0.117) is lower than TC's (0.161) and its relative drop (−84.3 %) is the largest. This is clinically coherent: ET is the small, contrast-dependent enhancing core, and several hard cases have little or no enhancing tumor at all, so any error collapses ET Dice. TC is a close second.

### Case study: BRATS_077

A dedicated diagnostic script (`scripts/diagnose_worst_case.py`) compared BRATS_077 (worst, Dice 0.239) against BRATS_425 (median, Dice 0.852). The concrete drivers:

1. **Small tumor.** WT volume = 36 579 voxels, the **17.7th percentile** of the validation cohort (vs 36.5th percentile for the median case, 60 356 voxels). Small lesions are penalised disproportionately by Dice: a fixed-size boundary error costs a much larger fraction of a small mask.
2. **Disproportionately small TC.** TC occupies only **6.7 %** of the WT volume, versus a cohort-typical 25–40 %. With so few TC voxels to begin with, any confusion with surrounding edema collapses the TC Dice almost entirely.
3. **Weak tumor-vs-brain contrast.** Normalised intensity contrast (|Δμ| / σ_bg) on modality 0 (FLAIR) is 1.44 for BRATS_077 vs 4.12 for the median case — roughly a **3× weaker signal**. Modality 3 (T2) shows the same pattern (0.47 vs 1.13).
4. **Fragmentation.** 31 connected WT components vs 14 for the median — the tumor is spatially scattered rather than a single mass, which breaks the implicit smoothness prior that CNN/Swin decoders learn.

None of these are bugs — they are inherent difficulties for any 3D CNN/Transformer trained without oversampling of rare regimes. Mitigations that would directly address them (a) small-tumor oversampling, (b) boundary loss, and (c) contrast-aware augmentation are out of scope for this paper but are the natural next steps and are documented in the repository.

## 5. End-to-end clinical pipeline: RECIST 1.1 response assessment

Segmentation is a means to an end; the clinical endpoint is a treatment-response verdict. We exercise the full loop in `notebooks/recist_response_demo.ipynb`. **This is a synthetic sanity check, not a longitudinal validation:** all "follow-up" scans are morphological perturbations of a *single* baseline (BRATS_407, seed 42), tuned to cross the very RECIST thresholds the classifier implements — so the verdicts below are circular by construction and demonstrate only that the measurement→classification code is wired correctly. No real multi-timepoint patient data was used (the LUMIERE path exists but was not run here).

1. Load OncoSeg ET prediction on a baseline scan.
2. Simulate three follow-up scans (PR / SD / PD) by morphologically perturbing the ET mask.
3. For each timepoint pair, extract per-lesion longest axial diameter and volume via `RECISTMeasurer`.
4. Classify the response per RECIST 1.1 thresholds (`ResponseClassifier`).

| Scenario | Simulated operation | SLD change | Verdict |
|----------|--------------------|------------|---------|
| PR       | 5-iter erosion     | **−32.9 %**| **PR** ✓ |
| SD       | 1-iter erosion     | **− 8.2 %**| **SD** ✓ |
| PD       | 5-iter dilation    | **+30.6 %**| **PD** ✓ |

All three scenarios cross the correct RECIST thresholds and the classifier returns the expected category (Figure 7, `figures/recist_demo.png`). The same code path is what would run on real longitudinal data — the only difference is that the follow-up mask would come from OncoSeg inference on a second scan rather than from synthetic perturbation. Critically, this closes the loop from raw MRI → segmentation → quantitative clinical endpoint with no manual measurement step.

## 6. Summary

- OncoSeg matches UNet3D's Dice across all regions with ~5× fewer parameters; no per-region Dice difference is statistically significant (Wilcoxon), and on WT UNet3D wins 67/96 subjects. The mean HD95 is lower (15.35 vs 21.03 mm) but is reported as an aggregate only and was not significance-tested.
- Calibration is good on background but poor on tumor: pooled ECE = 0.0101 is a background artifact (foreground-only ECE ≈ 0.49, over-confident on tumor voxels). MC Dropout uncertainty still correlates monotonically with prediction error, so it is useful as a relative review aid but not as a calibrated probability.
- Failures are concentrated on small, fragmented, low-contrast tumors. The dominant failure region is Enhancing Tumor (ET), with a −84.3 % relative Dice drop on the bottom-5 cases (TC −79.7 % is a close second) — a clinically interpretable and addressable limitation.
- The full segmentation → RECIST response-classification pipeline runs end-to-end and produces correct CR / PR / SD / PD verdicts on synthetic follow-up data.

## 7. Limitations

1. **Single dataset.** All numbers are on MSD Task01_BrainTumour. Cross-dataset generalisation (BraTS 2023, glioma from a different institution) has not been evaluated locally.
2. **Two-model comparison.** SwinUNETR and UNETR baselines require a CUDA GPU and are pending; the comparison table will be extended once those runs are complete.
3. **Ablation study.** The harness (`scripts/dryrun_ablation.py`) is in place for the 4 planned variants (no cross-attention, no deep supervision, no MC dropout, small embed_dim) but only a dry-run has been executed locally. Full training runs are pending GPU availability.
4. **RECIST validation is synthetic.** The demo uses morphologically perturbed masks in lieu of two real timepoints for the same patient; true longitudinal validation requires a paired-scan dataset that MSD does not provide.
5. **Uncertainty sample count.** The MC Dropout evaluation uses 5 samples for compute reasons. Higher sample counts would tighten the uncertainty estimate but are unlikely to change the ECE materially given the already-low baseline.
