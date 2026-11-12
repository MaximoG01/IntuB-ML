# AGENTS.md — istruzioni per assistenti AI e collaboratori

Leggi questo file prima di modificare il codice. Documenti di riferimento:
[SPEC.md](SPEC.md) (cosa), [ARCHITECTURE.md](ARCHITECTURE.md) (come), [DESIGN.md](DESIGN.md) (interfaccia).

## Contesto
- Progetto di ricerca/didattica: predizione della laringoscopia difficile (Cormack 3–4),
  soglia di rischio 15%. **Non è un dispositivo medico.**
- Il responsabile è un clinico, non uno sviluppatore: spiega le scelte tecniche in parole
  semplici e dai comandi pronti da incollare.
- Ambiente: Windows, PowerShell, Python 3.x, venv in `venv\`.
- Lingua: **interfaccia dell'app, pagine "Learn" e testi dei modelli esportati in inglese** (dal 03/11/2026); documenti di progetto (SPEC, README, ARCHITECTURE, DESIGN) in italiano; codice e nomi di variabili in inglese.

## Regole
1. **Mai committare dati**: niente `*.db`, export CSV, zip, `.env`, segreti. Controlla
   `git status` prima di ogni commit.
2. **Mai dati identificativi di pazienti** nel codice, nei test, nei log o negli esempi.
3. **Niente route che modificano o cancellano dati senza login, ruolo e CSRF.**
4. **Training separato dall'inferenza**: le route Flask non allenano modelli. Si allena in
   `training/`, si esporta in `models/`, l'app carica il JSON.
5. Ogni modello passa dalla stessa pipeline e dagli stessi fold; nessun passo di
   pre-elaborazione adattato su dati di validazione/test.
6. Valori mancanti = `None`/`NaN`, mai `0`.
7. Metriche: AUC con IC, calibrazione, metriche alla soglia 0,15, decision curve. Non usare
   l'accuracy come metrica principale.
8. I risultati sui **dati sintetici** vanno sempre etichettati come tali e non usati per
   affermazioni sulle prestazioni cliniche.
9. Prima di dire che qualcosa funziona: lancia i test (`pytest`) o l'app e dillo con l'output.
10. Modifiche piccole e motivate; aggiorna SPEC/ARCHITECTURE se cambi una decisione.
11. Non fare push, non cambiare la visibilità del repo e non riscrivere la storia git senza conferma.

## Comandi utili
```powershell
venv\Scripts\activate
$env:FLASK_APP = "wsgi.py"; flask run          # app in locale
flask db migrate -m "descrizione"; flask db upgrade   # migrazioni
python -m pytest                                # test (tests/)
python -m training.run_thyroid                 # studio tiroide (modello attivo): confronto + export in models/
python -m training.run_local_cases              # casi del sito: validazione + ricalibrazione (--export), non cambia ACTIVE
```
