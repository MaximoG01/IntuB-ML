# Model card — decassai-lr-v1

Created: 2026-11-02 · penalized logistic regression (L2, C=0.0464) · protocol: SPEC §11

## Data
- De Cassai et al., PLoS ONE 2019 (Padova, thyroid surgery; CC-BY 4.0), sha256 `1aed45c4f78f…`.
- 500 adults scheduled for thyroidectomy; 48 with Cormack 3–4 (9.6%), conventional curved blade.
- Patients with an expected difficult intubation excluded by protocol.
- Predictors: Mallampati III–IV, Mouth opening < 4 cm, Thyromental distance, Neck movement ≤ 90°, Cannot protrude the jaw, History of difficult intubation, BMI.

## Internal validation (stratified 5-fold CV × 20 repeats)
AUC computed within each fold; mean and 2.5–97.5% range across repeats (variability, not a CI).

| Model | AUC per fold | Brier | Calibr. slope | Sens. @15% | Spec. @15% | Flagged @15% |
|---|---|---|---|---|---|---|
| No model (prevalence) | 0.500 (0.500–0.500) | 0.087 (0.087–0.087) | 0.834 (0.834–0.834) | 0.0% (0.0%–0.0%) | 100.0% (100.0%–100.0%) | 0.0% (0.0%–0.0%) |
| El-Ganzouri ≥ 4 (recalibrated) | 0.597 (0.593–0.599) | 0.084 (0.084–0.085) | 1.060 (0.910–1.143) | 27.1% (27.1%–27.1%) | 92.3% (92.3%–92.3%) | 9.6% (9.6%–9.6%) |
| Mallampati ≥ III only | 0.635 (0.632–0.638) | 0.083 (0.082–0.084) | 1.064 (0.983–1.128) | 37.5% (37.5%–37.5%) | 89.6% (89.6%–89.6%) | 13.0% (13.0%–13.0%) |
| Penalized logistic (L2) | 0.738 (0.715–0.749) | 0.078 (0.077–0.081) | 1.119 (0.923–1.231) | 35.3% (32.2%–39.7%) | 89.0% (87.6%–89.8%) | 13.4% (12.4%–14.6%) |
| Gradient boosting | 0.676 (0.647–0.713) | 0.087 (0.084–0.092) | 0.458 (0.367–0.553) | 39.4% (32.2%–45.9%) | 83.0% (80.7%–85.2%) | 19.1% (17.2%–21.7%) |
| Neural network (MLP 8) | 0.687 (0.665–0.706) | 0.081 (0.079–0.083) | 0.779 (0.651–0.870) | 40.5% (34.3%–45.8%) | 86.0% (84.3%–87.6%) | 16.6% (14.8%–18.3%) |

**El-Ganzouri ≥ 4 as a fixed rule (M0)**: AUC 0.597, sensitivity 27.1%, specificity 92.3%, PPV 27.1%, flagged 9.6%.

## Optimism (Harrell's bootstrap)

| Model | Metric | Apparent | Optimism | Corrected |
|---|---|---|---|---|
| Penalized logistic (primary) | auc | 0.767 | 0.022 | 0.745 |
| Penalized logistic (primary) | brier | 0.075 | -0.003 | 0.078 |
| Penalized logistic (primary) | cal_slope | 1.318 | 0.129 | 1.189 |
| Unpenalized logistic | auc | 0.766 | 0.026 | 0.740 |
| Unpenalized logistic | brier | 0.074 | -0.004 | 0.078 |
| Unpenalized logistic | cal_slope | 1.001 | 0.188 | 0.812 |

## Coefficients (odds ratio per 1 standard deviation, bootstrap 95% CI)

- Mallampati III–IV: OR 1.34 (1.08–1.80)
- Mouth opening < 4 cm: OR 1.20 (0.96–1.55)
- Thyromental distance: OR 0.77 (0.49–0.92)
- Neck movement ≤ 90°: OR 1.05 (0.78–1.34)
- Cannot protrude the jaw: OR 0.86 (0.24–0.99)
- History of difficult intubation: OR 1.26 (1.09–1.87)
- BMI: OR 1.39 (1.19–2.10)

## Secondary analyses
- **+ neck circumference** (thyroid-only model): AUC per fold 0.764 (0.747–0.778), bootstrap-corrected 0.770.
- **VitalDB model on these patients**: AUC 0.635 (0.560–0.712); mean predicted risk 2.3% vs 9.6% observed.
- **Minimum sample size (Riley 2020)** with 7 predictors: 861 patients (83 events); available 500 (48).

## Risk of bias (PROBAST)

| Domain | Risk of bias | Why | Applicability (concern) | Why |
|---|---|---|---|---|
| 1. Participants | high | Patients with an expected difficult intubation were excluded by protocol: the most obvious cases are missing, and the ability of the predictors to discriminate is distorted (spectrum bias). | high | Thyroid surgery only (77% women, a quarter with a large goitre), one centre. |
| 2. Predictors | low | Measured at the anaesthesia visit, before intubation, with standard definitions. | low | Almost all the El-Ganzouri items, the same as in the app's form. |
| 3. Outcome | unclear | Cormack with a conventional curved blade, standard definition; the intubating clinician probably knows the pre-operative assessment (not blinded). | low | Exactly the project's outcome (direct laryngoscopy, Cormack 3–4). |
| 4. Analysis | high | 48 events for 7 predictors (about 7 per predictor), below Riley's sample size; Mallampati, neck and history arrive already cut into yes/no. In favour: pre-specified predictors, no missing values, penalization, calibration and optimism measured. | — | — |
| **Overall** | **high** | | **high** | |

## Limitations
- 48 events: estimates still unstable; sample size below Riley's criteria.
- Thyroid surgery only, one centre; patients with an expected difficult intubation excluded.
- Mallampati, neck movement and history of difficult intubation available only as yes/no.
- Overall risk of bias high (PROBAST). Not externally validated. **Do not use for clinical decisions.**

Source: De Cassai A et al., PLoS ONE 2019;14:e0212976, S1 Database (CC-BY 4.0).
