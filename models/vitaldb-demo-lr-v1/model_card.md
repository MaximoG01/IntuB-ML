# Model card — vitaldb-demo-lr-v1

Creato: 2026-10-31 · tipo: regressione logistica penalizzata (L2, C=0.0464)

## Dati
- VitalDB 1.0.0 (PhysioNet, CC-BY 4.0) — 5173 pazienti adulti (primo intervento per paziente), 131 con Cormack 3–4 (2.5%).
- Variabili: age, sex_male, bmi, asa, emergency. **Nessun esame delle vie aeree** (non presente in VitalDB).
- Popolazione coreana, chirurgia non cardiaca, un solo centro: la trasferibilità non è nota.
- Altezza e peso esclusi dopo una prima prova: insieme al BMI (che ne deriva) davano coefficienti opposti e instabili (collinearità). Le prestazioni erano uguali (AUC ~0,59).

## Validazione interna (CV stratificata 5 fold × 10 ripetizioni)
Valori: media e intervallo 2,5–97,5% tra le ripetizioni.

| Modello | AUC | AUPRC | Brier | Pendenza calibr. | Sens. @15% | VPP @15% | Segnalati @15% |
|---|---|---|---|---|---|---|---|
| Nessun modello (prevalenza) | 0.497 (0.497–0.497) | 0.025 (0.025–0.025) | 0.025 (0.025–0.025) | 0.930 (0.930–0.930) | 0.0% (0.0–0.0) | nan% (nan–nan) | 0.0% (0.0–0.0) |
| Logistica penalizzata (L2) | 0.596 (0.583–0.605) | 0.036 (0.034–0.038) | 0.025 (0.025–0.025) | 0.911 (0.775–0.932) | 0.0% (0.0–0.0) | nan% (nan–nan) | 0.0% (0.0–0.0) |
| Gradient boosting | 0.586 (0.573–0.614) | 0.035 (0.032–0.038) | 0.025 (0.025–0.025) | 0.426 (0.357–0.556) | 0.8% (0.0–1.5) | 7.7% (0.0–14.6) | 0.3% (0.2–0.3) |
| Rete neurale (MLP 8) | 0.613 (0.602–0.623) | 0.043 (0.039–0.045) | 0.025 (0.025–0.025) | 1.050 (0.955–1.178) | 0.0% (0.0–0.0) | nan% (nan–nan) | 0.0% (0.0–0.0) |

## Limiti
- Senza Mallampati, DTM, apertura bocca, mobilità del collo: il modello usa solo dati demografici.
- Il Cormack in VitalDB viene dalla cartella elettronica; non è noto se con lama diretta o videolaringoscopio.
- Non validato esternamente. **Non usare per decisioni cliniche.**

Fonte dati: Lee HC et al., VitalDB, Sci Data 2022 (CC-BY 4.0), https://physionet.org/content/vitaldb/1.0.0/
