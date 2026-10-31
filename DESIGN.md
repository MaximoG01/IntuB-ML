# DESIGN — intuB-ML (interfaccia)

Versione 0.1 · 31/10/2026 · stato: **bozza da confermare**

## 1. Cosa non funziona oggi

- La navigazione mescola cose per pubblici diversi: *Dashboard, Cases, Predict, Analytics, Blog*.
  Il clinico che deve valutare un paziente e lo studente che vuole capire la rete neurale vedono
  le stesse pagine.
- **Analytics** mette in fila confusion matrix, ROC, schema della rete, "loss landscape" e loss per
  epoca, senza dire cosa guardare né quale modello è in uso. Manca il grafico più importante per
  un rischio clinico: la **calibrazione**.
- Il "loss landscape" con traiettoria non rappresenta l'allenamento reale (vedi SPEC §8 #7).
- La pagina di predizione mostra un numero ma non spiega **perché** né quanto è affidabile.
- Testi misti inglese/italiano.

## 2. Pubblici e pagine

| Pubblico | Bisogno | Pagina |
|---|---|---|
| Clinico | valutare un paziente in 1 minuto, registrare l'esito | **Valuta**, **I miei casi** |
| Ricercatore | capire se il modello è affidabile, confrontare versioni | **Modello** |
| Studente / curioso | capire come funzionano logistica, rete, gradient descent | **Impara** (ex Blog + grafici didattici) |
| Admin | export dati, caricare modelli, utenti | **Admin** |

Menu: `Valuta · I miei casi · Modello · Impara` (+ `Admin` solo per admin).

## 3. Pagina "Valuta" (la più importante)

Form a sezioni, nell'ordine in cui si visita il paziente:
1. **Paziente**: età, sesso, peso, altezza (BMI calcolato e mostrato).
2. **Vie aeree (El-Ganzouri)**: apertura bocca, DTM, Mallampati, mobilità collo, prognatismo,
   storia di intubazione difficile. Sotto, il **punteggio EGRI calcolato in automatico** (0–12).
3. **Altro**: STOP-BANG (con aiuto alla compilazione).

Risultato in una scheda:
```
┌──────────────────────────────────────────────┐
│  Rischio di laringoscopia difficile           │
│  23 %   ███████░░░░░░░░░░  soglia 15 %        │
│  ▲ SOPRA SOGLIA — prepara piano B             │
│                                               │
│  EGRI 1996: 5 / 12 (positivo, ≥ 4)            │
│  Fattori che pesano di più:                   │
│   + DTM 5,8 cm      + Mallampati III           │
│   − apertura bocca 4,5 cm                      │
│                                               │
│  Modello egri-lr-v3 · dati sintetici ⚠        │
│  Non è un dispositivo medico.                 │
│  [Salva come caso in attesa]                  │
└──────────────────────────────────────────────┘
```
- Mostrare sempre **due numeri**: probabilità del modello e EGRI classico, così il clinico
  confronta con lo strumento che conosce.
- "Fattori che pesano di più": per la logistica, coefficiente × valore standardizzato.
- Etichetta ben visibile se il modello è allenato su dati sintetici.

## 4. Pagina "Modello"

In alto: nome e versione del modello attivo, data, numero di casi, % reali/sintetici.

Grafici, in quest'ordine, ognuno con una frase "come leggerlo":
1. **Calibrazione**: rischio previsto vs osservato (per decili), con diagonale. *È il primo
   grafico perché usiamo una soglia di probabilità.*
2. **ROC** di tutti i modelli confrontati (M0 EGRI incluso), con AUC e IC 95% in legenda;
   punto marcato alla soglia 15%.
3. **Decision curve**: beneficio netto vs soglia, con "prepara sempre" e "mai".
4. **Tabella alla soglia 15%**: sensibilità, specificità, VPP, VPN, % segnalati, per modello.
5. **Coefficienti** della logistica (odds ratio con IC), confrontati con i pesi EGRI.

Niente schema della rete né loss landscape qui: vanno in "Impara".

## 5. Pagina "Impara"

Contenuti didattici (ex Blog) + grafici che spiegano l'allenamento:
- Schema della rete con spessore/colore dei collegamenti = peso (quello attuale, migliorato).
- Loss per epoca, **training e validazione** sulla stessa figura (si vede l'overfitting).
- Gradient descent: esempio corretto su una **logistica a 2 variabili** (es. DTM e Mallampati),
  dove la superficie di loss si può calcolare davvero in 2D e la traiettoria è vera.

## 6. Stile

- Una sola lingua: **italiano** (eventuale inglese più avanti con file di traduzione).
- Colori: un colore per "sopra soglia" (arancio/rosso) e uno per "sotto soglia" (blu/verde);
  mai solo il colore per comunicare (aggiungere testo e icona). Contrasto AA.
- Una libreria grafica sola (Chart.js, già usato); stessi colori per lo stesso modello in tutti i grafici.
- Mobile first: il clinico usa spesso il telefono.
- Numeri con la virgola decimale e unità sempre visibili (cm, kg, %).
