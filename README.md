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

## Stato attuale (31/10/2026)

| Parte | Stato |
|---|---|
| App Flask (login, inserimento casi, predizione, analytics, blog) | funziona in locale |
| Dati | **nessun dato**: i dati sintetici sono stati abbandonati (circolari, vedi [SPEC.md](SPEC.md) §3.1). Prossima fonte: VitalDB (aperto) |
| Modelli | regressione logistica (scikit-learn); rete "distillata" 10→16→1 (PyTorch). Si allenano solo quando ci sono abbastanza casi reali |
| Deployment | era su Heroku (dismesso; alternative in [ARCHITECTURE.md](ARCHITECTURE.md)) |

Il database **non** è nel repo: in locale si crea vuoto con `flask db upgrade`.
Finché non ci sono almeno 20 casi con esito, l'app permette di inserire casi ma non mostra predizioni.

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

Apri http://127.0.0.1:5000. Per la rete neurale serve anche `pip install torch` (solo per l'allenamento in locale).

## Riferimento principale

el-Ganzouri AR et al. *Preoperative airway assessment: predictive value of a multivariate risk index.*
Anesth Analg 1996;82:1197–204. [doi:10.1097/00000539-199606000-00017](https://doi.org/10.1097/00000539-199606000-00017)
