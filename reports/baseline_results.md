# Rezultate baseline (CP2)

Generat de `scripts/run_baselines.py`. Metricile sunt pe clasa alunecare, din confuzia cumulată pe toți pixelii valizi. Pragurile și hiperparametrii sunt fixați pe train; testul a fost rulat o singură dată, cu configurația fixată.

- **CV G1 (fold 1–4):** benzi din grila G1, cu restul G1 la antrenare: generalizare în același sit (medie ± abatere standard între fold-uri).
- **Fold 5 (G2–G5):** grilele mici, cu antrenare aproape doar pe G1: transfer către altă zonă.
- **Validare / test:** situri oficiale noi, model antrenat pe tot train-ul.

## Tabelul principal: F1 (%)

| Baseline | Reprezentare | CV G1 (fold 1–4) | Fold 5 (G2–G5) | CV 5 fold-uri | Validare | Test | IoU test | AUC test | AP test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Totul negativ | - | 0,00 ± 0,00 | 0,00 | 0,00 ± 0,00 | 0,00 | 0,00 | 0,00 | 0,500 | 0,020 |
| Prag NDVI | A: S2 + indici + pantă + DEM | 27,74 ± 8,21 | 16,68 | 25,53 ± 8,66 | 35,15 | 34,81 | 21,07 | 0,941 | 0,249 |
| Prag pantă | B: pantă + DEM | 6,92 ± 1,77 | 2,69 | 6,08 ± 2,43 | 4,34 | 5,55 | 2,85 | 0,581 | 0,030 |
| RF pe pixeli (A) | A: S2 + indici + pantă + DEM | 68,09 ± 2,61 | 11,19 | 56,71 ± 25,55 | 46,09 | 46,59 | 30,37 | 0,963 | 0,442 |
| RF pe pixeli (B) | B: pantă + DEM | 5,30 ± 0,68 | 1,50 | 4,54 ± 1,80 | 3,39 | 3,90 | 1,99 | 0,565 | 0,023 |

Referință din literatură: U-Net, test oficial (Ghorbanzadeh et al., 2022b), F1 = 59,92%. U-Net folosește contextul spațial al patch-ului; baseline-urile de aici clasifică fiecare pixel separat.

## Precizie și recall (%)

| Baseline | Evaluare | Precizie | Recall | Prag |
|---|---|---:|---:|---:|
| Prag NDVI | validare | 22,59 | 79,14 | 0,4053 |
| Prag NDVI | test | 22,66 | 75,05 | 0,4053 |
| Prag pantă | validare | 2,87 | 8,86 | 2,02 |
| Prag pantă | test | 3,71 | 11,01 | 2,02 |
| RF pe pixeli (A) | validare | 43,31 | 49,24 | 0,7525 |
| RF pe pixeli (A) | test | 43,54 | 50,10 | 0,7525 |
| RF pe pixeli (B) | validare | 2,10 | 8,87 | 0,2688 |
| RF pe pixeli (B) | test | 2,45 | 9,63 | 0,2688 |

Pragul se aplică scorului: −NDVI la pragul NDVI, panta la pragul de pantă, probabilitatea la RF.

## F1 pe fold (%)

| Baseline | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| Totul negativ | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| Prag NDVI | 37,38 | 27,24 | 28,97 | 17,37 | 16,68 |
| Prag pantă | 7,74 | 4,61 | 6,61 | 8,73 | 2,69 |
| RF pe pixeli (A) | 67,09 | 67,99 | 71,71 | 65,58 | 11,19 |
| RF pe pixeli (B) | 5,95 | 4,53 | 4,94 | 5,78 | 1,50 |

Configurație RF: 100 arbori, adâncime maximă 20, minimum 5 pixeli în frunză, seed 42; eșantion de antrenare: până la 50 pixeli de alunecare și 150 de fundal per patch.
