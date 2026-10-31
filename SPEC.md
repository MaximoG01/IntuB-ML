# SPEC — intuB-ML

Versione 0.1 · 31/10/2026 · stato: **bozza da confermare**.
Le voci marcate **[DA DECIDERE]** sono scelte aperte; accanto c'è la proposta.

---

## 1. Problema clinico

Prevedere prima dell'induzione se la **laringoscopia diretta sarà difficile**, per preparare
in anticipo videolaringoscopio, bougie, secondo operatore o strategia da sveglio.

- **Esito (outcome)**: `difficile = 1` se Cormack-Lehane **3 o 4** alla laringoscopia; `0` se 1–2.
  - Nota: l'indice originale di El-Ganzouri (1996) è stato costruito sul **solo grado IV**
    (107 casi su 10.507, circa 1%). Con 3–4 la prevalenza attesa è più alta (ordine del 5–10%,
    variabile per popolazione); i pesi originali non sono quindi tarati sul nostro esito.
  - **[DA DECIDERE]** registrare a parte anche: numero di tentativi, uso di presidi
    (bougie, VL), fallimento. Proposta: sì, come esiti secondari; l'esito principale resta Cormack 3–4.
  - Va sempre registrato **con quale strumento** è stato dato il Cormack (lama Macintosh vs VL):
    il grado con videolaringoscopio non è confrontabile.
- **Soglia di rischio**: caso segnalato se `P(difficile) ≥ 0,15`.
  - È una **soglia decisionale**, non una soglia "di calibrazione": vuol dire "preparo il piano B
    se il rischio è almeno 1 su 7". Ha senso solo se il modello è **calibrato** (vedi §6).
  - La soglia va giustificata con la **decision curve analysis** (§6) e resa configurabile.

## 2. Variabili di ingresso

### 2.1 Indice di El-Ganzouri (EGRI) — 7 item, punteggio 0–12, positivo se ≥ 4

| Item | 0 | 1 | 2 | Nel DB oggi? |
|---|---|---|---|---|
| Apertura bocca (distanza interincisiva) | ≥ 4 cm | < 4 cm | — | sì (`dii`, continuo) |
| Distanza tiromentoniera | > 6,5 cm | 6,0–6,5 cm | < 6,0 cm | sì (`dtm`, continuo) |
| Mallampati | I | II | III | sì (`mallampati`, 1–4) |
| Mobilità del collo | > 90° | 80–90° | < 80° | **no** |
| Prognatismo possibile | sì | no | — | **no** |
| Peso | < 90 kg | 90–110 kg | > 110 kg | sì (`weight`) |
| Storia di intubazione difficile | no | dubbia | certa | **no** |

(Verificare la tabella sul lavoro originale prima di usarla nel codice.)

### 2.2 Altre variabili già presenti
età, sesso, altezza (→ BMI), STOP-BANG (rischio OSAS), farmaco, tecnica.

### 2.3 Requisiti sulle variabili
- Salvare **gli item grezzi** (misure in cm, gradi, classi), non solo il totale EGRI.
  Il totale si calcola dal codice. Oggi `alganzouri` è un numero inserito a mano **e**
  i suoi componenti sono anch'essi input: doppio conteggio e incoerenze possibili.
- Aggiungere al modello dati: `neck_movement`, `prognathism`, `history_difficult`,
  `cormack_device`, `attempts`, `adjunct_used`. Facoltativi: circonferenza collo, test del morso del labbro superiore.
- Valori mancanti = `NULL`, **mai 0**. Oggi `/predict` trasforma un campo vuoto in `0`
  (`height=0`, `dtm=0`), che il modello legge come valore reale: è un bug.

## 3. Dati

### 3.1 Dati sintetici — ABBANDONATI (31/10/2026)
Decisione: i dati sintetici non si usano più, e i generatori sono stati rimossi dal codice.
Generare casi dalla formula di El-Ganzouri e poi validarci sopra un modello restituisce la formula
stessa: è un risultato circolare, e le metriche sono fuorvianti. Resta qui la diagnosi, come memoria.

- `app/ml.py::_synthetic_seed` (rimosso) generava una variabile latente da età, BMI, DTM e DII; da quella
  derivano Mallampati, STOP-BANG, "alganzouri" e Cormack, con poco rumore.
- Risultato misurato (31/10/2026, 2.000 casi, CV a 5 fold): prevalenza **56%**, AUC della
  logistica **0,999**, AUC del solo "alganzouri" **0,996**. Il generatore è quasi deterministico
  e non riproduce l'indice El-Ganzouri vero (mancano 3 item su 7).
- Esistono **tre generatori diversi** (`seed_synthetic.py`, `app/seed_realistic.py`,
  `seed_proper.py`). `seed_realistic.py`, usato dagli script di deploy Heroku, estrae il Cormack
  **a caso, indipendente dalle variabili**: con quei dati nessun modello può imparare nulla.

### 3.2 Modello di partenza senza dati sintetici
Il "modello precaricato" è l'EGRI stesso scritto come logistica (M0/M1, §4): pesi presi dal
lavoro del 1996, dichiarati come tali. Si aggiorna solo con dati reali (prima ricalibrazione,
poi stima di pesi nuovi).

### 3.3 Dati reali — dove cercarli (ricerca del 31/10/2026)
Non risultano dataset aperti con Cormack **e** esame completo delle vie aeree (Mallampati, DTM...).

| Fonte | Accesso | Cosa c'è | Uso possibile |
|---|---|---|---|
| **VitalDB** (Seoul National Univ. Hospital), su PhysioNet | **aperto**, CC-BY 4.0 | 6.388 interventi; `cormack` per 5.553: I 5.178, II 235, IIIa 91, IIIb 35, IV 14 → **140 casi 3–4 (2,5%)**; età, sesso, altezza, peso, BMI, ASA, tipo di intervento, esami pre-op. **Nessun Mallampati** né altro esame vie aeree | modello "solo dati demografici" come confronto; studio della prevalenza; test della pipeline su dati veri |
| **MIMIC-IV** (PhysioNet) | credenziali + corso CITI | terapia intensiva; intubazioni presenti; Cormack/Mallampati probabilmente solo nelle note libere | da verificare (MIMIC-IV-Note) |
| R `medicaldata::laryngoscope` | aperto | RCT ~100 obesi, Pentax AWS vs Macintosh | solo didattico |
| Studi pubblicati (es. Frontiers Public Health 2022, dati su BioStudies) | da verificare | variabili demografiche, ML a confronto | da verificare |
| Database nazionali (danese DAD, MPOG USA) | su richiesta/consorzio | grandi coorti con valutazione vie aeree | richiesta formale di collaborazione |

Strade, in ordine di fattibilità:
1. **VitalDB subito** (aperto): pipeline e modello demografico su 5.553 casi reali.
2. **Raccolta prospettica propria** con l'app: serve comitato etico, informativa privacy e dati
   **pseudonimizzati** (GDPR art. 9).
3. **Dati individuali da autori** di studi di validazione EGRI; contatti con DAD/MPOG.
4. **MIMIC-IV**: richiedere le credenziali e verificare se le note contengono il Cormack.

Il file VitalDB si scarica in `data/external/vitaldb/` (ignorato da git; licenza CC-BY: citare la fonte).

### 3.4 Dimensione campionaria
Con un esito raro servono molti casi. Riferimento: criteri di Riley et al. (BMJ 2020) per modelli
di predizione. Ordine di grandezza: con ~10 variabili e prevalenza 7% servono **migliaia** di casi
(centinaia di eventi). Una rete neurale ne richiede ancora di più.

## 4. Modelli

Tutti i modelli producono `P(difficile)` e passano dalla **stessa** pipeline e dagli stessi fold.

| ID | Modello | Ruolo |
|---|---|---|
| M0 | EGRI ≥ 4 (pesi fissi 1996) | **riferimento clinico**: ogni modello va confrontato con questo |
| M1 | Logistica su punteggio EGRI (ricalibrazione: intercetta + pendenza) | aggiornamento minimo |
| M2 | Logistica penalizzata (L2 o elastic net) sugli item grezzi, con spline per le continue | **candidato principale** |
| M3 | Gradient boosting (es. `HistGradientBoostingClassifier`) | controllo non lineare |
| M4 | Rete neurale piccola (MLP, 1–2 strati nascosti) | solo se i dati reali sono abbastanza |

Note sui modelli esistenti:
- **La logistica non va cambiata, va migliorata**: penalizzazione, spline, calibrazione,
  niente riallenamento a ogni richiesta `/predict` (oggi succede).
- La **"distilled NN"** attuale non usa i pesi della logistica: impara a imitare le probabilità
  della logistica (insegnante) con una rete più grande (studente 10→16→1). Di solito si distilla
  da un modello grande a uno piccolo; qui è il contrario, quindi la rete al massimo ricopia la
  logistica. Proposta: tenerla come esempio didattico, non come modello in produzione.
- La rete non è "pesante" in uso: 10×16 pesi si calcolano in numpy in microsecondi. È pesante
  **l'allenamento con PyTorch**, che va fatto solo offline (vedi ARCHITECTURE.md).
- Su dati tabellari piccoli, la logistica penalizzata è in genere competitiva con modelli più
  complessi ed è più facile da spiegare: è il candidato da battere.

## 5. Pipeline (tutto dentro un `sklearn.Pipeline`, adattato **solo** sul training di ogni fold)

1. Pulizia: tipi, range plausibili, unità, duplicati.
2. Mancanti: imputazione (mediana o multipla) **dentro** la pipeline + indicatori di mancanza.
3. Trasformazioni: spline per le continue; codifica one-hot o ordinale per le classi.
4. Standardizzazione (per logistica e rete; inutile per il boosting).
5. Regolarizzazione: forza scelta con CV **annidata** (nested CV), non sul set di test.
6. Calibrazione: verificare; se serve, Platt/isotonica con CV.
7. Soglia 0,15 fissata **prima** di guardare i risultati.

## 6. Validazione e metriche

Confrontare più algoritmi è **metodologicamente corretto** se:
- stessi dati, stessi fold (CV stratificata ripetuta, es. 5×10), stessa pipeline;
- metriche e confronto stabiliti **prima** (pre-registrazione nel README dei risultati);
- nessuna informazione del test entra nel training (scaler, imputazione, selezione variabili);
- si riportano intervalli di confidenza (bootstrap) e non solo il "vincitore";
- si segue una checklist: **TRIPOD+AI** (2024) per i modelli di predizione clinica.
Molte review/repo con "10 algoritmi a confronto" non rispettano questi punti (leakage, sola
accuracy, nessuna calibrazione): vanno lette con questo filtro.

Metriche da riportare per ogni modello:
- **Discriminazione**: AUC ROC con IC 95%; AUC precision-recall (utile con esito raro).
- **Calibrazione**: grafico di calibrazione, intercetta e pendenza, Brier score.
- **Alla soglia 0,15**: sensibilità, specificità, VPP, VPN, % pazienti segnalati.
- **Utilità clinica**: decision curve analysis (beneficio netto vs "prepara sempre" / "mai").
- **Non usare l'accuracy** come metrica principale: con esito raro è ingannevole.

Validazione esterna (altro centro o altro periodo) prima di qualsiasi uso clinico.

## 7. Requisiti dell'app

Funzionali:
- R1 Inserimento valutazione pre-intubazione (tutti gli item §2) → probabilità + spiegazione.
- R2 Salvataggio come caso "in attesa" e completamento con l'esito dopo l'intubazione.
- R3 Pagina "Modello": metriche §6, versione del modello, data, dati usati (sintetici/reali).
- R4 Esportazione dati pseudonimizzati per l'allenamento offline (solo utenti autorizzati).
- R5 Caricamento di una nuova versione del modello senza modificare il codice.

Non funzionali:
- N1 Nessun dato identificativo del paziente (nome, data di nascita, n. cartella) nel DB.
- N2 Nessun database, export o segreto nel repo git.
- N3 Nessuna route che modifica dati senza login e senza CSRF; niente endpoint "segreti".
- N4 Predizione < 200 ms senza PyTorch sul server.
- N5 Avviso visibile "non è un dispositivo medico" in ogni pagina di predizione.

## 8. Problemi aperti nel codice (da risolvere in quest'ordine)

| # | Problema | Dove | Gravità |
|---|---|---|---|
| 1 | ✅ 31/10/2026 — `app.db`, `db_export/`, `intub-export.zip` tolti dal repo e da `.gitignore` | radice | — |
| 2 | ✅ 31/10/2026 — rimosso l'endpoint `/seed-db-secret-key-12345` (cancellava tutti i casi senza login) | `app/routes.py` | — |
| 3 | ✅ 31/10/2026 — `SECRET_KEY` solo da variabile d'ambiente (altrimenti casuale) | `config.py` | — |
| 4 | ✅ Campi vuoti non più convertiti in 0; il caso si salva anche senza modello | `app/routes.py::predict` | — |
| 5 | ✅ La predizione usa il modello esportato (`app/inference.py`); nessun training nelle route di predizione | `app/routes.py` | — |
| 6 | ✅ Stessa soglia 0,15 per logistica e rete | `app/ml_nn.py` | — |
| 7 | ✅ Rimosso il "loss landscape" con traiettoria (proiettava pesi della rete su componenti PCA dei dati: senza significato) | `analytics` | — |
| 8 | ✅ Rimossi generatori sintetici e fallback; senza dati l'app lo dice | `app/ml.py` | — |
| 9 | ✅ Rimossi script di debug/test e deploy Heroku; vecchi .md in `docs/legacy/` | radice | — |
| 10 | ✅ Rimosso `wsl_import.sh` con **password del DB Heroku in chiaro**; storia git ripulita | radice | — |
| 11 | `app/nn_viz.py` e `app/ml_nn.py` non più usati dalle pagine (da spostare in `training/` o in "Impara") | `app/` | bassa |
| 12 | Punti Mallampati EGRI: le fonti secondarie non concordano (I=0, II=1, III–IV=2 usato qui) → verificare sul lavoro originale | `app/egri.py` | media |

## 9. Risultati — VitalDB (31/10/2026)

5.173 pazienti adulti (primo intervento), 131 Cormack 3–4 (2,5%). Variabili: età, sesso, BMI, ASA, urgenza.
CV stratificata 5 fold × 10 ripetizioni, stessa pipeline per tutti.

| Modello | AUC | Pendenza calibrazione | Segnalati @15% |
|---|---|---|---|
| Nessun modello (prevalenza) | 0,50 | — | 0% |
| **Logistica penalizzata (attivo)** | **0,60** (0,58–0,61) | 0,91 | 0% |
| Gradient boosting | 0,59 (0,57–0,61) | 0,43 | 0,3% |
| Rete neurale MLP 8 | 0,61 (0,60–0,62) | 1,05 | 0% |

Conclusioni:
- Con i soli dati demografici la discriminazione è scarsa con **qualunque** algoritmo: il limite
  sono le variabili, non il modello. La rete neurale non aggiunge nulla di rilevante.
- Con prevalenza 2,5% nessun paziente supera il 15%: la soglia ha senso solo con predittori forti
  (esame delle vie aeree). Rivalutare la soglia con la decision curve quando ci saranno quei dati.
- Odds ratio per 1 DS: età 1,25 · sesso maschile 1,24 · BMI 1,15 · ASA 0,94 · urgenza 0,94.
- Prima prova con altezza + peso + BMI: coefficienti opposti e instabili (collinearità) → tenuto solo il BMI.
