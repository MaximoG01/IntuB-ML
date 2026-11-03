# Model card — laryngoscope-lr-v1

Creato: 2026-11-02 · tipo: regressione logistica penalizzata (L2, C=0.00599) · protocollo: SPEC §10

## Dati
- medicaldata::laryngoscope (Abdallah et al., Anesth Analg 2011; licenza MIT), commit `be27ec8`, sha256 `539476b20d14…`.
- 99 adulti obesi, chirurgia elettiva; 18 con Cormack 3–4 (18.2%).
- Esito: `view == 0` (la documentazione del pacchetto ha la codifica invertita; vedi SPEC §10).
- Predittori: mallampati, bmi, age, sex_male, video_laryngoscope. Nell'app il videolaringoscopio è fissato a 0 (predizione per lama Macintosh).

## Validazione interna (CV stratificata 5 fold × 20 ripetizioni)
Media e intervallo 2,5–97,5% tra le ripetizioni (variabilità della CV, **non** un IC). AUC calcolata dentro ogni fold: sulle predizioni out-of-fold riunite, con pochi eventi, i modelli deboli scendono artificialmente sotto 0,5 (il modello nullo dava 0,31). Per lo stesso motivo la pendenza di calibrazione della CV è falsata per i modelli quasi costanti: per la calibrazione vale il bootstrap qui sotto.

| Modello | AUC per fold | Brier | Pendenza calibr. | Sens. @15% | Spec. @15% | Segnalati @15% |
|---|---|---|---|---|---|---|
| Nessun modello (prevalenza) | 0.500 (0.500–0.500) | 0.149 (0.149–0.149) | -4.148 (-4.148–-4.148) | 100.0% (100.0%–100.0%) | 0.0% (0.0%–0.0%) | 100.0% (100.0%–100.0%) |
| Solo Mallampati | 0.470 (0.338–0.555) | 0.152 (0.149–0.158) | -1.310 (-2.992–0.290) | 74.7% (55.6%–91.8%) | 15.1% (5.5%–22.9%) | 83.0% (74.6%–93.9%) |
| Logistica penalizzata (L2) | 0.544 (0.451–0.602) | 0.152 (0.148–0.158) | -0.642 (-1.760–0.647) | 76.9% (66.7%–97.4%) | 13.5% (3.6%–22.2%) | 84.8% (76.2%–96.1%) |
| Gradient boosting | 0.558 (0.472–0.638) | 0.160 (0.143–0.173) | 0.184 (-0.064–0.555) | 50.6% (41.5%–58.5%) | 57.8% (51.2%–64.2%) | 43.7% (37.3%–49.0%) |
| Rete neurale (MLP 8) | 0.492 (0.370–0.560) | 0.158 (0.149–0.165) | -0.215 (-0.836–0.327) | 54.4% (41.2%–64.0%) | 30.1% (25.3%–38.4%) | 67.1% (57.9%–71.7%) |

## Ottimismo (bootstrap di Harrell)

| Modello | Metrica | Apparente | Ottimismo | Corretto |
|---|---|---|---|---|
| Logistica penalizzata (principale) | auc | 0.629 | 0.088 | 0.540 |
| Logistica penalizzata (principale) | brier | 0.147 | -0.012 | 0.159 |
| Logistica penalizzata (principale) | cal_slope | 9.115 | 0.889 | 8.226 |
| Logistica senza penalizzazione | auc | 0.635 | 0.103 | 0.532 |
| Logistica senza penalizzazione | brier | 0.138 | -0.020 | 0.158 |
| Logistica senza penalizzazione | cal_slope | 1.000 | 0.485 | 0.516 |

La pendenza corretta del modello penalizzato non è interpretabile: le sue predizioni sono quasi costanti, e una piccola differenza tra pazienti produce una pendenza grande. Il modello senza penalizzazione mostra invece l'overfitting tipico: pendenza corretta molto sotto 1.

## Coefficienti (odds ratio per 1 deviazione standard, IC 95% bootstrap)

- Mallampati: OR 1.01 (0.65–1.56)
- BMI: OR 1.00 (0.70–1.38)
- Età: OR 1.02 (0.77–1.83)
- Sesso maschile: OR 1.04 (1.00–2.04)
- Videolaringoscopio: OR 0.98 (0.45–1.09)

## Analisi secondarie
- **Sola lama Macintosh** (n = 49, 11 eventi): AUC per fold 0.583 (0.526–0.670).
- **Modello VitalDB su questi pazienti**: AUC 0.562 (0.398–0.726); rischio medio previsto 3.7% contro 18.2% osservato; segnalati @15% 0.0%.
- **Numerosità minima (Riley 2020)** con 5 predittori e prevalenza 18%: 465 pazienti (85 eventi); disponibili 99 (18).

## Rischio di bias (PROBAST)

| Dominio | Rischio di bias | Perché | Applicabilità (preoccupazione) | Perché |
|---|---|---|---|---|
| 1. Partecipanti | incerto | Dati di un RCT (fonte accettabile), ma criteri di esclusione da verificare sull'articolo originale: se escludeva vie aeree difficili note, la prevalenza è distorta. | alta | Solo obesi (BMI 30–50), chirurgia elettiva, un centro: non rappresenta la popolazione generale della domanda (§1). |
| 2. Predittori | basso | Predittori misurati prima della laringoscopia, definizioni standard, scelti prima dell'analisi. | alta | Manca quasi tutto l'El-Ganzouri (DTM, apertura bocca, collo, prognatismo, storia): c'è solo il Mallampati. |
| 3. Esito | alto | Cormack dato da chi intuba, che conosce il Mallampati (non in cieco); metà dei pazienti valutata con videolaringoscopio, che migliora la visione: l'esito non è uguale per tutti. | alta | La domanda è la laringoscopia diretta; qui l'esito è solo binario (1–2 vs 3–4). |
| 4. Analisi | alto | 18 eventi per 5 predittori (3,6 eventi per predittore), molto sotto la numerosità minima di Riley. A favore: predittori pre-specificati, niente selezione univariata, mancanti imputati dentro i fold, penalizzazione, calibrazione e ottimismo misurati. | — | — |
| **Complessivo** | **alto** | | **alta** | |

## Limiti
- Solo 18 eventi: stime instabili, intervalli di confidenza larghi.
- Solo obesi in chirurgia elettiva, un centro: non trasferibile ad altre popolazioni senza validazione.
- Unico esame delle vie aeree disponibile: Mallampati. El-Ganzouri non calcolabile.
- Metà dei Cormack con videolaringoscopio: il modello lo usa come predittore; nell'app è fissato a lama Macintosh.
- Rischio di bias complessivo alto (PROBAST). Non validato esternamente. **Non usare per decisioni cliniche.**

Fonte: Abdallah et al., Anesth Analg 2011;113:1082-7; Nowacki AS, TSHS Resources Portal 2017; pacchetto R medicaldata (MIT).
