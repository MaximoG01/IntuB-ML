# intuB-ML — Predizione della laringoscopia difficile

> **Prototipo di ricerca e didattica. Non è un dispositivo medico.**
> Non usarlo per decisioni cliniche reali. I dati attuali sono **sintetici**.

intuB-ML stima, prima dell'induzione, la **probabilità che la laringoscopia sia difficile**
(Cormack-Lehane 3–4) a partire da dati di valutazione delle vie aeree raccolti al letto del paziente.
Se la probabilità supera il **15%** il caso viene segnalato come "a rischio".

Il progetto ha due obiettivi:
1. **Raccolta casi**: un'app web dove il clinico inserisce la valutazione pre-intubazione e, dopo, l'esito.
2. **Modello che si aggiorna**: partire dall'indice di El-Ganzouri (pesi fissi, 1996) e arrivare a
   pesi stimati sui dati raccolti, confrontando modelli diversi in modo corretto.

## Stato attuale (02/11/2026)

| Parte | Stato |
|---|---|
| App Flask (login, inserimento casi, predizione, pagina Modello, sezione Impara) | funziona in locale |
| Dati | **Chirurgia tiroidea, Padova** (De Cassai et al., PLoS ONE 2019: 500 pazienti, 48 Cormack 3–4, item di El-Ganzouri) per il modello attivo ([SPEC.md](SPEC.md) §11). Studio sugli obesi (99 pazienti, §10) e VitalDB (solo dati demografici, §9) come confronto. Dati sintetici abbandonati (§3.1) |
| Modello attivo | `thyroid-lr-v1`: logistica penalizzata su 7 item (Mallampati, apertura bocca, DTM, collo, prognatismo, storia, BMI). **AUC ≈ 0,74** contro 0,60 della regola El-Ganzouri ≥ 4; rischio di bias alto (PROBAST), solo chirurgia tiroidea |
| Confronto | stessi fold per tutti: El-Ganzouri 0,60, Mallampati 0,64, logistica 0,74, gradient boosting 0,68, rete neurale 0,69. Sugli obesi nessun modello supera 0,56 |
| El-Ganzouri | calcolato dall'app dai 7 item, mostrato accanto al modello |
| Deployment | Heroku dismesso; alternative in [ARCHITECTURE.md](ARCHITECTURE.md) |

Il database **non** è nel repo: in locale si crea vuoto con `flask db upgrade`.

## Documentazione

| File | Contenuto |
|---|---|
| [SPEC.md](SPEC.md) | cosa deve fare il sistema: esito, variabili, dati, modelli, validazione, requisiti |
| [ARCHITECTURE.md](ARCHITECTURE.md) | come è fatto oggi e come lo vogliamo: training offline, app online, hosting |
| [DESIGN.md](DESIGN.md) | interfaccia: pagine, grafici, testi |
| [AGENTS.md](AGENTS.md) | regole per chi lavora sul codice con un assistente AI (Claude Code, Codex, Copilot) |

## Avvio in locale (Windows, PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
$env:FLASK_APP = "wsgi.py"
flask db upgrade
flask run
```

Apri http://127.0.0.1:5000.

### Condividere sulla rete locale (colleghi sulla stessa Wi-Fi/LAN)

```powershell
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\start_local_network.ps1
```

La prima volta chiede un **codice invito** e crea `.env.lan` (segreti, ignorato da git). Poi mostra
l'indirizzo da dare ai colleghi (es. `http://192.168.1.20:8000`): si registrano con il codice invito.
Se Windows chiede l'accesso alla rete, consenti solo le reti **private**. Solo casi senza dati
identificativi: per pazienti reali servono comitato etico e informativa privacy (SPEC §3.3).

Per nominare l'amministratore (aggiunge variabili ed esporta i casi), dopo che si è registrato:

```powershell
$env:FLASK_APP = "wsgi.py"; flask make-admin nome@esempio.it
```

### Variabili aggiuntive

L'amministratore apre **Variables** nel menu e aggiunge una variabile (numero con unità e
intervallo ammesso, oppure scelta da una lista): compare subito nel form **Assess** per tutti.
Non entra nel calcolo del rischio finché non si allena un modello che la usa; i valori si scaricano
con **Download CSV** (colonne `x_<nome>`, cella vuota = mancante) per l'allenamento offline.

### Allenare e confrontare i modelli (offline)

```powershell
pip install -r requirements-train.txt
python -m training.run_thyroid      # studio tiroide Padova (SPEC §11): confronta i modelli, esporta models/thyroid-lr-v1/ e lo rende attivo
python -m training.run_laryngoscope  # studio sugli obesi (SPEC §10), esporta models/laryngoscope-lr-v1/
python -m training.run_vitaldb       # (vecchio) modello solo demografico su VitalDB
python -m pytest                     # test
```

Il risultato va in `models/<nome>/model.json` (pesi + metriche + dati dei grafici) e
`model_card.md`; `models/ACTIVE` dice quale modello usa l'app. Il server non allena nulla.

## Riferimento principale

el-Ganzouri AR et al. *Preoperative airway assessment: predictive value of a multivariate risk index.*
Anesth Analg 1996;82:1197–204. [doi:10.1097/00000539-199606000-00017](https://doi.org/10.1097/00000539-199606000-00017)
