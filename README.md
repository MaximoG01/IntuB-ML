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
| Dati | 2.000 casi **sintetici** + pochi casi di test; nessun dato reale |
| Modelli | regressione logistica (scikit-learn); rete "distillata" 10→16→1 (PyTorch) |
| Deployment | era su Heroku (ora da sostituire, vedi [ARCHITECTURE.md](ARCHITECTURE.md)) |

⚠️ Problemi noti più importanti (dettagli in [SPEC.md](SPEC.md)):
- I dati sintetici sono quasi deterministici (AUC ≈ 0,999 per qualunque modello): servono solo a
  provare il codice, **non** a dire quale modello è migliore.
- Prevalenza sintetica 56% di casi difficili; nella realtà Cormack 3–4 è molto più raro.
- Nessun dato reale: le metriche mostrate dall'app non dicono nulla sulle prestazioni cliniche.

Il database **non** è nel repo: in locale si crea con `flask db upgrade` e si popola con uno
script di seed (vedi SPEC §3).

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
