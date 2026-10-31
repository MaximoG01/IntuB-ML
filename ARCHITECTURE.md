# ARCHITECTURE — intuB-ML

Versione 0.1 · 31/10/2026 · stato: **bozza da confermare**

## 1. Com'è oggi

```
Browser ──► Flask (app/routes.py, un solo file da ~600 righe)
              ├─ login / registrazione (Flask-Login, Flask-WTF)
              ├─ /predict  → allena la logistica da capo a ogni richiesta → probabilità
              ├─ /analytics → allena di nuovo logistica (+ rete se c'è PyTorch),
              │               loss surface, SVG della rete
              ├─ /cases, /case/new, /case/<id>/complete
              └─ (fino al 31/10/2026: /seed-db-secret-key-12345, rimosso)
            SQLAlchemy ──► SQLite (app.db) in locale / Postgres su Heroku
            app/data/trained_model.json  ← pesi della rete allenata in locale (240 KB, con storico pesi)
```

Problemi di architettura:
- Training e inferenza sono **mescolati** nelle route: lento, non riproducibile (il modello cambia
  quando cambia il DB) e non si sa quale modello ha prodotto una predizione.
- Nessuna versione dei modelli; nessun legame tra predizione salvata e modello usato.
- Dati sintetici e reali nella **stessa tabella**, senza un campo che li distingua.

## 2. Come lo vogliamo

Principio: **si allena in locale, online si usa e basta.**

```
┌──────────────── LOCALE (PC del ricercatore) ────────────────┐
│ training/                                                   │
│   data_synth.py   generatore sintetico v2 (seme fisso)      │
│   pull_data.py    scarica i casi pseudonimizzati dall'API   │
│   pipeline.py     pulizia → imputazione → spline → scaler   │
│   compare.py      M0–M4, CV ripetuta, metriche SPEC §6      │
│   export.py       salva il modello scelto in models/        │
│ models/egri-lr-v3/                                          │
│   model.json      coefficienti / pesi (niente pickle)       │
│   model_card.md   dati, metriche, limiti, data              │
│   plots/*.json    dati dei grafici (ROC, calibrazione, DCA) │
└──────────────────────────┬──────────────────────────────────┘
                           │ git push (o upload da pagina admin)
┌──────────────── ONLINE (server leggero) ────────────────────┐
│ Flask                                                        │
│   app/inference.py  carica models/<attivo>/model.json        │
│                     predizione in numpy, < 1 ms              │
│   app/routes/       predict · cases · model · admin · auth   │
│   /api/export       CSV pseudonimizzato (solo admin, token)  │
│ Postgres gestito: users, cases (+ data_source, model_version)│
└──────────────────────────────────────────────────────────────┘
```

Il ciclo di aggiornamento:
1. I clinici inseriscono casi ed esiti online.
2. Il ricercatore in locale lancia `pull_data.py` → `compare.py` → sceglie il modello.
3. `export.py` scrive un nuovo `models/<nome>-vN/`; si fa commit e push.
4. Il server ricarica il modello; ogni nuova predizione salva `model_version`.

Così il server **non ha bisogno di PyTorch né di scikit-learn per allenare**: solo numpy per
calcolare logistica o MLP dai pesi in JSON. Le immagini dei grafici sono generate dal browser
(Chart.js) a partire dai JSON in `plots/`.

## 3. Flask o Streamlit?

| | Flask (attuale) | Streamlit |
|---|---|---|
| Login multiutente, ruoli | sì (già fatto) | limitato (o servizio esterno) |
| Inserimento casi, form, DB | sì (già fatto) | possibile ma scomodo |
| Grafici e esplorazione | serve JavaScript (Chart.js) | molto semplice |
| Hosting gratuito | vari (sotto) | Streamlit Community Cloud |

**Proposta**: restare su **Flask** per l'app (raccolta dati + predizione: login e DB sono già
fatti). Usare eventualmente **Streamlit solo in locale** come "laboratorio" per esplorare dati e
confrontare modelli. **[DA DECIDERE]**

## 4. Hosting al posto di Heroku

Opzioni comuni per un'app Flask piccola (prezzi e limiti cambiano spesso: **verificarli** prima di scegliere):

| Servizio | Note |
|---|---|
| **Render** | deploy da GitHub come Heroku; piano gratuito che si "addormenta" dopo inattività |
| **PythonAnywhere** | pensato per Flask/Django; piano gratuito limitato, a pagamento economico |
| **Google Cloud Run** | container, si paga a richieste; quota gratuita mensile; serve Docker |
| **Fly.io / Railway** | container, piani a pagamento economici |
| DB: **Neon** o **Supabase** | Postgres gestito con piano gratuito; separare DB e server |

**Proposta**: per la demo con dati sintetici → Render (o PythonAnywhere) + Neon/Supabase.
**Con dati di pazienti reali** la scelta cambia: server e DB in UE, contratto di trattamento
dati (DPA), approvazione del DPO/comitato etico; probabilmente infrastruttura dell'ospedale o
dell'università. **[DA DECIDERE]**

## 5. Struttura del repo (proposta)

```
intuB-ML/
├─ README.md  SPEC.md  ARCHITECTURE.md  DESIGN.md  AGENTS.md  CLAUDE.md
├─ app/                  # online
│   ├─ __init__.py  config.py  models.py  inference.py
│   ├─ routes/  (auth.py, predict.py, cases.py, model.py, admin.py)
│   ├─ templates/  static/
├─ training/             # offline
├─ models/               # modelli esportati (JSON + model card)
├─ migrations/
├─ tests/                # pytest: pipeline, inferenza = training, route
├─ docs/legacy/          # i vecchi *_SUMMARY.md, *_GUIDE.md
└─ requirements.txt  requirements-train.txt
```

`requirements.txt` (server): Flask e estensioni, numpy, gunicorn, psycopg.
`requirements-train.txt` (locale): scikit-learn, pandas, matplotlib, torch (facoltativo).

## 6. Sicurezza e privacy (vincoli di architettura)

- `.gitignore`: `*.db`, `db_export/`, `*.zip`, `.env`. Rimuovere dal repo e **dalla storia**
  i file già committati (es. `git filter-repo`), poi cambiare le password degli utenti presenti.
- Segreti (`SECRET_KEY`, `DATABASE_URL`) solo come variabili d'ambiente.
- Nessun identificativo del paziente; ID del caso casuale (UUID), non progressivo.
- Ruoli: `clinician` (inserisce casi propri), `admin` (export, caricamento modelli).
