# Model card — decassai-lr-v1

Creato: 2026-11-02 · regressione logistica penalizzata (L2, C=0.0464) · protocollo: SPEC §11

## Dati
- De Cassai et al., PLoS ONE 2019 (Padova, chirurgia tiroidea; CC-BY 4.0), sha256 `1aed45c4f78f…`.
- 500 adulti per tiroidectomia; 48 con Cormack 3–4 (9.6%), lama curva convenzionale.
- Esclusi per protocollo i pazienti con intubazione difficile prevista.
- Predittori: Mallampati III–IV, Apertura bocca < 4 cm, Distanza tiromentoniera, Mobilità del collo ≤ 90°, Prognatismo non possibile, Storia di intubazione difficile, BMI.

## Validazione interna (CV stratificata 5 fold × 20 ripetizioni)
AUC calcolata dentro ogni fold; media e intervallo 2,5–97,5% tra le ripetizioni (variabilità, non IC).

| Modello | AUC per fold | Brier | Pendenza calibr. | Sens. @15% | Spec. @15% | Segnalati @15% |
|---|---|---|---|---|---|---|
| Nessun modello (prevalenza) | 0.500 (0.500–0.500) | 0.087 (0.087–0.087) | 0.834 (0.834–0.834) | 0.0% (0.0%–0.0%) | 100.0% (100.0%–100.0%) | 0.0% (0.0%–0.0%) |
| El-Ganzouri ≥ 4 (ricalibrato) | 0.597 (0.593–0.599) | 0.084 (0.084–0.085) | 1.060 (0.910–1.143) | 27.1% (27.1%–27.1%) | 92.3% (92.3%–92.3%) | 9.6% (9.6%–9.6%) |
| Solo Mallampati ≥ III | 0.635 (0.632–0.638) | 0.083 (0.082–0.084) | 1.064 (0.983–1.128) | 37.5% (37.5%–37.5%) | 89.6% (89.6%–89.6%) | 13.0% (13.0%–13.0%) |
| Logistica penalizzata (L2) | 0.738 (0.715–0.749) | 0.078 (0.077–0.081) | 1.119 (0.923–1.231) | 35.3% (32.2%–39.7%) | 89.0% (87.6%–89.8%) | 13.4% (12.4%–14.6%) |
| Gradient boosting | 0.676 (0.647–0.713) | 0.087 (0.084–0.092) | 0.458 (0.367–0.553) | 39.4% (32.2%–45.9%) | 83.0% (80.7%–85.2%) | 19.1% (17.2%–21.7%) |
| Rete neurale (MLP 8) | 0.687 (0.665–0.706) | 0.081 (0.079–0.083) | 0.779 (0.651–0.870) | 40.5% (34.3%–45.8%) | 86.0% (84.3%–87.6%) | 16.6% (14.8%–18.3%) |

**El-Ganzouri ≥ 4 come regola fissa (M0)**: AUC 0.597, sensibilità 27.1%, specificità 92.3%, VPP 27.1%, segnalati 9.6%.

## Ottimismo (bootstrap di Harrell)

| Modello | Metrica | Apparente | Ottimismo | Corretto |
|---|---|---|---|---|
| Logistica penalizzata (principale) | auc | 0.767 | 0.022 | 0.745 |
| Logistica penalizzata (principale) | brier | 0.075 | -0.003 | 0.078 |
| Logistica penalizzata (principale) | cal_slope | 1.318 | 0.129 | 1.189 |
| Logistica senza penalizzazione | auc | 0.766 | 0.026 | 0.740 |
| Logistica senza penalizzazione | brier | 0.074 | -0.004 | 0.078 |
| Logistica senza penalizzazione | cal_slope | 1.001 | 0.188 | 0.812 |

## Coefficienti (odds ratio per 1 deviazione standard, IC 95% bootstrap)

- Mallampati III–IV: OR 1.34 (1.08–1.80)
- Apertura bocca < 4 cm: OR 1.20 (0.96–1.55)
- Distanza tiromentoniera: OR 0.77 (0.49–0.92)
- Mobilità del collo ≤ 90°: OR 1.05 (0.78–1.34)
- Prognatismo non possibile: OR 0.86 (0.24–0.99)
- Storia di intubazione difficile: OR 1.26 (1.09–1.87)
- BMI: OR 1.39 (1.19–2.10)

## Analisi secondarie
- **+ circonferenza del collo** (modello solo per la tiroide): AUC per fold 0.764 (0.747–0.778), corretta con bootstrap 0.770.
- **Modello VitalDB su questi pazienti**: AUC 0.635 (0.560–0.712); rischio medio previsto 2.3% contro 9.6% osservato.
- **Numerosità minima (Riley 2020)** con 7 predittori: 861 pazienti (83 eventi); disponibili 500 (48).

## Rischio di bias (PROBAST)

| Dominio | Rischio di bias | Perché | Applicabilità (preoccupazione) | Perché |
|---|---|---|---|---|
| 1. Partecipanti | alto | Esclusi per protocollo i pazienti con intubazione difficile prevista: mancano proprio i casi più evidenti, e la capacità dei predittori di distinguere è distorta (bias di spettro). | alta | Solo chirurgia tiroidea (77% donne, un quarto con gozzo voluminoso), un centro. |
| 2. Predittori | basso | Misurati alla visita anestesiologica, prima dell'intubazione, con definizioni standard. | bassa | Sono quasi tutti gli item di El-Ganzouri, gli stessi del form dell'app. |
| 3. Esito | incerto | Cormack con lama curva convenzionale, definizione standard; chi intuba probabilmente conosce la valutazione pre-operatoria (non in cieco). | bassa | È esattamente l'esito del progetto (laringoscopia diretta, Cormack 3–4). |
| 4. Analisi | alto | 48 eventi per 7 predittori (circa 7 per predittore), sotto la numerosità di Riley; Mallampati, collo e storia arrivano già tagliati in sì/no. A favore: predittori pre-specificati, nessun mancante, penalizzazione, calibrazione e ottimismo misurati. | — | — |
| **Complessivo** | **alto** | | **alta** | |

## Limiti
- 48 eventi: stime ancora instabili; numerosità sotto i criteri di Riley.
- Solo chirurgia tiroidea, un centro; esclusi i pazienti con intubazione difficile prevista.
- Mallampati, mobilità del collo e storia di intubazione difficile disponibili solo come sì/no.
- Rischio di bias complessivo alto (PROBAST). Non validato esternamente. **Non usare per decisioni cliniche.**

Fonte: De Cassai A et al., PLoS ONE 2019;14:e0212976, S1 Database (CC-BY 4.0).
