# IC3 · H2 · Versant: susceptibilitate la alunecări

Mini-twin pentru detecția alunecărilor din Sentinel-2 și DEM/pantă, cu un proxy topografic de susceptibilitate, pe setul Landslide4Sense.

Descrierea datelor: [`docs/datacard_H2.md`](docs/datacard_H2.md).

## Instalare

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Date

Datele nu sunt în repo. Imaginile și măștile de train se descarcă de pe [Kaggle](https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense) și se pun astfel:

```text
data/
├── TrainData/img/image_N.h5
├── TrainData/mask/mask_N.h5
├── ValidData/img/image_N.h5
├── ValidData/mask/mask_N.h5
├── TestData/img/image_N.h5
└── TestData/mask/mask_N.h5
```

Măștile de validare și test se descarcă de pe Hugging Face cu:

```bash
python scripts/download_masks.py
```

## Rulare

Din rădăcina repo-ului:

```bash
python scripts/check_data.py
python scripts/check_structure.py
python scripts/check_mask_match.py
python scripts/plot_patch.py --split train --id 241
```

Rezultatele ajung în `reports/`. Seed-ul implicit este 42.

## Structură

| Cale | Conținut |
|---|---|
| `src/l4s/` | Citirea patch-urilor |
| `scripts/` | Pașii care se rulează |
| `reports/` | Statistici și figuri generate |
| `docs/` | Data card și documente de proiect |
| `data/` | Setul de date, ignorat de Git |
