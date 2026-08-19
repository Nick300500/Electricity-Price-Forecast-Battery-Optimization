# Handoff / Status

Stand: 2026-08-19, Branch `Modularisierung` (noch nicht in `main` gemerged).

## Was bisher passiert ist

1. **Modularisierung** — Notebook (74 Zellen, top-to-bottom, viel Dopplung) in `src/battery_opt/` zerlegt: `optimizer.py`, `data_loading.py`, `data_prep.py`, `modeling.py`, `backtest.py`, `plotting.py`. Notebook orchestriert nur noch.
2. **Environment** — lokales venv `.venv/` (nicht in git), als Jupyter-Kernel `battery-optimization` / Anzeigename "Battery Optimization" registriert.
3. **Terminal-getriebene Pipeline** — `scripts/run_pipeline.py` → `battery_opt/pipeline.py`. Läuft komplett ohne Notebook, ~15-20s. Schreibt:
   - `Results/battery_signal_report_2025.csv` (time, P_ch, P_dis, E, predicted_price, actual_price, profit)
   - `Results/battery_cumulative_profit_2025.csv` (aufsummierter Profit alle 24h)
4. **Inhaltliche Fixes aus dem Review** (`Notebook review.md`):
   - Data-Leakage im RF-Fit gefixt (fit nur auf X_train/y_train)
   - GridSearchCV nutzt jetzt `TimeSeriesSplit` statt plain KFold (+ `n_jobs=-1` parallelisiert, 14min → 4min)
   - Wind/Solar/Other-Generation-Features kommen jetzt aus SMARD-Day-Ahead-**Prognosen** statt Ist-Werten (`battery_opt/smard_client.py`, automatischer Fetch falls fehlend in `Data/`)
     - Dabei einen Bug im Community-OpenAPI-Spec gefunden: PV-Filter ist **125**, nicht 126 (Enum vs. Beschreibung widersprüchlich im Spec) — empirisch verifiziert (Tag/Nacht-Kurve + Summe = total_forecast)

## Offener Punkt: net_load / res_load

Bleiben noch auf **Ist-Werten** (nicht Prognose) — das ist der letzte "Actuals statt Forecast"-Punkt aus dem Review.

- SMARDs **neue** Download-Center-Oberfläche (`/home/downloadcenter/download-marktdaten/`) nutzt ein anderes, undokumentiertes Backend-Schema (`superCategoryId`/`subcategoryId`, POST an `/nip-download-manager/nip/download/market-data`). Ich hab ~90 Requests reverse-engineert (Feldnamen gefunden: `format`, `region`, `categories`, `language`, `timestamp_from`, `timestamp_to`, `type`, `resolution`), aber komme nicht über einen "No value present"-Fehler hinweg (vermutlich fehlende interne Übersetzungs-Lookup). Abgebrochen, nicht mehr produktiv.
- Offizielles SMARD-Benutzerhandbuch (PDF) bestätigt: "Prognostizierter Stromverbrauch" wird eigentlich von der **ENTSO-E Transparency Platform** bezogen (Document Type A65, Tagesprognose Gesamtlast).
- **Plan:** ENTSO-E hat eine offiziell dokumentierte REST-API, aber Zugang braucht Registrierung + E-Mail an transparency@entsoe.eu ("RESTful API access"), Freischaltung dauert laut deren Doku ~3 Werktage.

### Nächste Schritte (in der Reihenfolge, wie besprochen)

1. **User**: Auf https://transparency.entsoe.eu registrieren, Freischaltungs-Mail schicken (falls noch nicht passiert).
2. **Zwischenlösung**: User lädt SMARD-CSV für "Prognostizierter Stromverbrauch" manuell runter (Download-Button auf der Seite, keine API nötig), legt sie in `Data/` ab (z.B. `load_forecast_2023.csv` etc.) — dann schreibe ich einen Loader dafür, analog zu den anderen manuell geladenen Dateien.
3. **Sobald ENTSO-E-Token da ist**: `battery_opt/entsoe_client.py` bauen (analog zu `smard_client.py`), `data_prep.py` von der manuellen CSV auf den automatisierten Client umstellen. Damit läuft dann auch `net_load`/`res_load` automatisiert wie die Generation-Forecasts.

## Sonstige offene Punkte aus dem Review (noch nicht angefasst)

- Kein Rolling/Walk-Forward-Retraining (der eigentliche "rolling aufbauen"-Wunsch des Users) — nächster großer Schritt nach dem Feature-Realismus-Thema.
- Kohlepreis ist monatlich, flach auf jede Stunde gebroadcastet — Feature-Importance-Check steht noch aus (niedrige Priorität).
- Ziel: dauerhaft laufendes System, das täglich neu vorhersagt / rollend nachtrainiert (aktuell nur Backtest auf H1 2025).

## Umgebung / wie weiterarbeiten

```
.venv\Scripts\python.exe scripts\run_pipeline.py          # ganze Pipeline
.venv\Scripts\python.exe scripts\fetch_forecast_data.py   # nur SMARD-Gen-Forecasts nachladen
```

`Data/` (Rohdaten) ist nicht in git — liegt aber lokal vor. `requirements.txt` ist aktuell.
