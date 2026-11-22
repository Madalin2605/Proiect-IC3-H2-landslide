# IC3 · H2 · Versant: susceptibilitate la alunecări

Mini-twin pentru detecția alunecărilor din Sentinel-2 și DEM/pantă, cu un proxy topografic de susceptibilitate, pe setul Landslide4Sense. Clientul este un consiliu județean care vrea să prioritizeze verificările în teren.

**Stadiu: CP3 (Învățare).** Modelul v1 (U-Net) a fost comparat cu baseline-urile din CP2 pe 3 seed-uri, cu intervale de încredere de 95%.

| Model (F1 pe clasa alunecare, %) | CV G1 | Fold 5 (altă zonă) | Validare | Test oficial |
|---|---:|---:|---:|---:|
| **U-Net, A** | **73,3** [71,8; 74,6] | **30,2** [20,6; 37,7] | **57,3** [44,1; 67,9] | **59,4** [55,8; 63,3] |
| RF pe pixeli, A (cel mai bun baseline) | 68,2 [66,6; 69,6] | 11,3 [8,1; 15,5] | 44,8 [32,7; 54,4] | 45,3 [42,1; 48,9] |
| U-Net, B (doar topografie) | 7,2 [6,5; 7,9] | 4,3 [2,9; 5,4] | 4,5 [2,5; 6,8] | 5,2 [4,0; 6,5] |

A = Sentinel-2 + indici + pantă + DEM; B = doar pantă + DEM. U-Net-ul bate RF pe toate evaluările, cu intervalul diferenței peste zero peste tot. Transferul către o zonă nouă (fold 5) rămâne slab. Reprezentarea B dă un rezultat negativ: nu se distinge de un prag simplu pe pantă. Detalii în [`docs/results_H2.md`](docs/results_H2.md).

## Documente

| Document | Etapa | Conținut |
|---|---|---|
| [`docs/datacard_H2.md`](docs/datacard_H2.md) | CP1 | Descrierea datelor, verificări, limite |
| [`docs/soa_H2.md`](docs/soa_H2.md) | CP1 | Stadiul actual (literatură) |
| [`docs/methods_H2.md`](docs/methods_H2.md) | CP2 | Reprezentările A și B, protocolul cu fold-uri spațiale, baseline-urile |
| [`docs/results_H2.md`](docs/results_H2.md) | CP3 | Modelul v1 față de baseline-uri, pe seed-uri, cu intervale |

## Instalare

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Pentru CP3 e nevoie și de PyTorch cu CUDA. Nu vine de pe PyPI, se instalează separat (testat pe Python 3.12, driver CUDA 12.1, GTX 1650 de 4 GB):

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
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

Toate comenzile se rulează din rădăcina repo-ului. Fiecare etapă își scrie rezultatele în `reports/CP1/`, `reports/CP2/` sau `reports/CP3/`.

**CP1: datele**

```bash
python scripts/check_data.py
python scripts/check_structure.py
python scripts/check_mask_match.py
python scripts/plot_patch.py --split train --id 241
```

**CP2: reprezentarea, protocolul și baseline-urile**

```bash
python scripts/describe_representations.py   # reprezentările A și B, AUC pe trăsătură
python scripts/make_splits.py                # 5 fold-uri spațiale pe train, cu buffer
python scripts/run_baselines.py              # totul negativ, praguri, RF pe A și B, seed 42 (~20 de minute)
```

Pentru o verificare rapidă a baseline-urilor: `--folds 1 --skip-test`.

**CP3: modelul v1 față de baseline-uri, pe 3 seed-uri**

```bash
python scripts/build_cache.py                # cache NumPy pentru rețele (~2,9 GB în data/cache/) + verificări
python scripts/train_unet.py --rep s2_topo   # U-Net pe A: 3 seed-uri x (5 fold-uri + tot train-ul), ~4 ore pe GPU
python scripts/train_unet.py --rep topo      # U-Net pe B, ~4 ore pe GPU
python scripts/run_rf_seeds.py               # RF și praguri pe aceleași seed-uri și fold-uri, ~1 oră pe procesor
python scripts/aggregate_results.py          # intervale, comparații pereche, tabele, figură (câteva secunde)
```

Antrenarea și RF-ul pot fi oprite și repornite: rulările terminate nu se refac. Rezultatele pe rulare (`patches.csv`, `run.json`, `hist.npz`) sunt în repo, deci `aggregate_results.py` se poate rula fără reantrenare. Greutățile modelelor (`model.pt`) sunt ignorate de Git.

## Structură

| Cale | Conținut |
|---|---|
| `src/l4s/` | Citirea patch-urilor, trăsături, metrici, fold-uri spațiale, cache și U-Net |
| `scripts/` | Pașii care se rulează |
| `reports/CP1/` | Statistici despre date și verificări (CP1) |
| `reports/CP2/` | Reprezentări, fold-uri (`splits.json`), rezultatele baseline (CP2) |
| `reports/CP3/` | `results_v1_H2.csv`, tabelele din `results_v1.md`, figura și rulările din `runs/` (CP3) |
| `docs/` | Data card și secțiunile articolului |
| `data/` | Setul de date și cache-ul, ignorate de Git |
