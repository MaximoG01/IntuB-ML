# Model card — laryngoscope-lr-v1

Created: 2026-11-02 · type: penalized logistic regression (L2, C=0.00599) · protocol: SPEC §10

## Data
- medicaldata::laryngoscope (Abdallah et al., Anesth Analg 2011; MIT licence), commit `be27ec8`, sha256 `539476b20d14…`.
- 99 obese adults, elective surgery; 18 with Cormack 3–4 (18.2%).
- Outcome: `view == 0` (the package documentation has the coding inverted; see SPEC §10).
- Predictors: mallampati, bmi, age, sex_male, video_laryngoscope. In the app the video laryngoscope is fixed to 0 (prediction for a Macintosh blade).

## Internal validation (stratified 5-fold CV × 20 repeats)
Mean and 2.5–97.5% range across repeats (CV variability, **not** a CI). AUC computed within each fold: on pooled out-of-fold predictions, with few events, weak models fall artificially below 0.5 (the null model gave 0.31). For the same reason the CV calibration slope is distorted for near-constant models: for calibration use the bootstrap below.

| Model | AUC per fold | Brier | Calibr. slope | Sens. @15% | Spec. @15% | Flagged @15% |
|---|---|---|---|---|---|---|
| No model (prevalence) | 0.500 (0.500–0.500) | 0.149 (0.149–0.149) | -4.148 (-4.148–-4.148) | 100.0% (100.0%–100.0%) | 0.0% (0.0%–0.0%) | 100.0% (100.0%–100.0%) |
| Mallampati only | 0.470 (0.338–0.555) | 0.152 (0.149–0.158) | -1.310 (-2.992–0.290) | 74.7% (55.6%–91.8%) | 15.1% (5.5%–22.9%) | 83.0% (74.6%–93.9%) |
| Penalized logistic (L2) | 0.544 (0.451–0.602) | 0.152 (0.148–0.158) | -0.642 (-1.760–0.647) | 76.9% (66.7%–97.4%) | 13.5% (3.6%–22.2%) | 84.8% (76.2%–96.1%) |
| Gradient boosting | 0.558 (0.472–0.638) | 0.160 (0.143–0.173) | 0.184 (-0.064–0.555) | 50.6% (41.5%–58.5%) | 57.8% (51.2%–64.2%) | 43.7% (37.3%–49.0%) |
| Neural network (MLP 8) | 0.492 (0.370–0.560) | 0.158 (0.149–0.165) | -0.215 (-0.836–0.327) | 54.4% (41.2%–64.0%) | 30.1% (25.3%–38.4%) | 67.1% (57.9%–71.7%) |

## Optimism (Harrell's bootstrap)

| Model | Metric | Apparent | Optimism | Corrected |
|---|---|---|---|---|
| Penalized logistic (primary) | auc | 0.629 | 0.088 | 0.540 |
| Penalized logistic (primary) | brier | 0.147 | -0.012 | 0.159 |
| Penalized logistic (primary) | cal_slope | 9.115 | 0.889 | 8.226 |
| Unpenalized logistic | auc | 0.635 | 0.103 | 0.532 |
| Unpenalized logistic | brier | 0.138 | -0.020 | 0.158 |
| Unpenalized logistic | cal_slope | 1.000 | 0.485 | 0.516 |

The corrected slope of the penalized model cannot be interpreted: its predictions are almost constant, and a small difference between patients produces a large slope. The unpenalized model shows typical overfitting instead: corrected slope well below 1.

## Coefficients (odds ratio per 1 standard deviation, bootstrap 95% CI)

- Mallampati: OR 1.01 (0.65–1.56)
- BMI: OR 1.00 (0.70–1.38)
- Age: OR 1.02 (0.77–1.83)
- Male sex: OR 1.04 (1.00–2.04)
- Video laryngoscope: OR 0.98 (0.45–1.09)

## Secondary analyses
- **Macintosh blade only** (n = 49, 11 events): AUC per fold 0.583 (0.526–0.670).
- **VitalDB model on these patients**: AUC 0.562 (0.398–0.726); mean predicted risk 3.7% vs 18.2% observed; flagged @15% 0.0%.
- **Minimum sample size (Riley 2020)** with 5 predictors and prevalence 18%: 465 patients (85 events); available 99 (18).

## Risk of bias (PROBAST)

| Domain | Risk of bias | Why | Applicability (concern) | Why |
|---|---|---|---|---|
| 1. Participants | unclear | RCT data (an acceptable source), but the exclusion criteria must be checked in the original paper: if known difficult airways were excluded, the prevalence is distorted. | high | Only obese patients (BMI 30–50), elective surgery, one centre: not the general population of our question (§1). |
| 2. Predictors | low | Predictors measured before laryngoscopy, standard definitions, chosen before the analysis. | high | Almost all of El-Ganzouri is missing (TMD, mouth opening, neck, jaw protrusion, history): only Mallampati is available. |
| 3. Outcome | high | Cormack graded by the intubating clinician, who knows the Mallampati (not blinded); half of the patients assessed with a video laryngoscope, which improves the view: the outcome is not the same for everyone. | high | Our question is direct laryngoscopy; here the outcome is only binary (1–2 vs 3–4). |
| 4. Analysis | high | 18 events for 5 predictors (3.6 events per predictor), far below Riley's minimum sample size. In favour: pre-specified predictors, no univariable selection, missing values imputed within the folds, penalization, calibration and optimism measured. | — | — |
| **Overall** | **high** | | **high** | |

## Limitations
- Only 18 events: unstable estimates, wide confidence intervals.
- Only obese patients in elective surgery, one centre: not transferable to other populations without validation.
- The only airway examination available is Mallampati. El-Ganzouri cannot be computed.
- Half of the Cormack grades were obtained with a video laryngoscope: the model uses the device as a predictor; in the app it is fixed to a Macintosh blade.
- Overall risk of bias high (PROBAST). Not externally validated. **Do not use for clinical decisions.**

Source: Abdallah et al., Anesth Analg 2011;113:1082-7; Nowacki AS, TSHS Resources Portal 2017; R package medicaldata (MIT).
