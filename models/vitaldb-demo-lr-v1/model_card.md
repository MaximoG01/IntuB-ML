# Model card — vitaldb-demo-lr-v1

Created: 2026-10-31 · type: penalized logistic regression (L2, C=0.0464)

## Data
- VitalDB 1.0.0 (PhysioNet, CC-BY 4.0) — 5173 adult patients (first surgery per patient), 131 with Cormack 3–4 (2.5%).
- Variables: age, sex_male, bmi, asa, emergency. **No airway examination** (not available in VitalDB).
- Korean population, non-cardiac surgery, one centre: transferability unknown.
- Height and weight excluded after a first try: together with BMI (derived from them) they gave opposite, unstable coefficients (collinearity). Performance was the same (AUC ~0.59).

## Internal validation (stratified 5-fold CV × 10 repeats)
Values: mean and 2.5–97.5% range across repeats.

| Model | AUC | AUPRC | Brier | Calibr. slope | Sens. @15% | PPV @15% | Flagged @15% |
|---|---|---|---|---|---|---|---|
| No model (prevalence) | 0.497 (0.497–0.497) | 0.025 (0.025–0.025) | 0.025 (0.025–0.025) | 0.930 (0.930–0.930) | 0.0% (0.0–0.0) | nan% (nan–nan) | 0.0% (0.0–0.0) |
| Penalized logistic (L2) | 0.596 (0.583–0.605) | 0.036 (0.034–0.038) | 0.025 (0.025–0.025) | 0.911 (0.775–0.932) | 0.0% (0.0–0.0) | nan% (nan–nan) | 0.0% (0.0–0.0) |
| Gradient boosting | 0.586 (0.573–0.614) | 0.035 (0.032–0.038) | 0.025 (0.025–0.025) | 0.426 (0.357–0.556) | 0.8% (0.0–1.5) | 7.7% (0.0–14.6) | 0.3% (0.2–0.3) |
| Neural network (MLP 8) | 0.613 (0.602–0.623) | 0.043 (0.039–0.045) | 0.025 (0.025–0.025) | 1.050 (0.955–1.178) | 0.0% (0.0–0.0) | nan% (nan–nan) | 0.0% (0.0–0.0) |

## Limitations
- No Mallampati, TMD, mouth opening or neck movement: the model uses demographics only.
- The Cormack grade in VitalDB comes from the electronic record; whether a direct blade or a video laryngoscope was used is unknown.
- Not externally validated. **Do not use for clinical decisions.**

Data source: Lee HC et al., VitalDB, Sci Data 2022 (CC-BY 4.0), https://physionet.org/content/vitaldb/1.0.0/
