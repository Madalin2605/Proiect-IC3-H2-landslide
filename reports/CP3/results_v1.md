# Rezultate v1 (CP3): model vs. baseline, pe seed-uri, cu intervale

Generat de `scripts/aggregate_results.py` din `reports/CP3/runs/`. Seed-uri: 42, 43, 44. F1 pe clasa alunecare, din confuzia cumulată pe pixelii valizi; valoarea = media pe seed-uri (la CV G1: pe fold-urile 1-4 x seed-uri). În paranteze: interval 95% bootstrap (1000 reeșantionări; în train pe rânduri de grilă, în validare/test pe patch-uri).

## Tabelul principal: F1 (%) [IC 95%]

| Model | CV G1 (fold 1-4) | Fold 5 (G2-G5) | Validare | Test oficial | AP test |
|---|---:|---:|---:|---:|---:|
| U-Net (A) | 73,3 [71,8, 74,6] | 30,2 [20,6, 37,7] | 57,3 [44,1, 67,9] | 59,4 [55,8, 63,3] | 0,666 |
| RF pe pixeli (A) | 68,2 [66,6, 69,6] | 11,3 [8,1, 15,5] | 44,8 [32,7, 54,4] | 45,3 [42,1, 48,9] | 0,433 |
| Prag NDVI (A) | 27,7 [25,7, 29,9] | 16,7 [12,1, 20,8] | 34,9 [21,6, 47,2] | 34,6 [29,0, 40,7] | 0,249 |
| U-Net (B) | 7,2 [6,5, 7,9] | 4,3 [2,9, 5,4] | 4,5 [2,5, 6,8] | 5,2 [4,0, 6,5] | 0,028 |
| RF pe pixeli (B) | 5,5 [5,1, 5,8] | 1,5 [1,1, 2,0] | 3,4 [2,3, 4,6] | 4,0 [3,3, 4,7] | 0,023 |
| Prag pantă (B) | 6,9 [6,4, 7,5] | 2,8 [2,0, 3,5] | 4,2 [2,8, 6,0] | 5,4 [4,1, 6,9] | 0,030 |

Referință: U-Net din literatură pe testul oficial, F1 = 59,9%. „Totul negativ” are F1 = 0.

## Variabilitatea între seed-uri: F1 (%) min-max

| Model | CV G1 (fold 1-4) | Fold 5 (G2-G5) | Validare | Test oficial |
|---|---:|---:|---:|---:|
| U-Net (A) | 72,6-73,8 | 28,7-32,3 | 56,5-58,9 | 58,5-61,3 |
| RF pe pixeli (A) | 68,1-68,3 | 11,1-11,7 | 43,7-46,1 | 43,6-47,0 |
| Prag NDVI (A) | 27,7-27,7 | 16,6-16,8 | 34,6-35,1 | 34,4-34,8 |
| U-Net (B) | 7,0-7,4 | 2,8-6,0 | 4,2-4,8 | 5,0-5,3 |
| RF pe pixeli (B) | 5,4-5,6 | 1,5-1,6 | 3,4-3,5 | 4,0-4,0 |
| Prag pantă (B) | 6,9-7,0 | 2,7-3,0 | 4,1-4,3 | 5,2-5,5 |

La CV G1, min-max e între mediile pe fold-uri ale fiecărui seed; abaterea standard pe cele 12 rulări: U-Net (A) 1,6; RF pe pixeli (A) 2,3; Prag NDVI (A) 7,4; U-Net (B) 1,5; RF pe pixeli (B) 0,7; Prag pantă (B) 1,5.

## Precizie, recall, AUC, AP (medii pe seed-uri)

| Model | Evaluare | Precizie | Recall | IoU | AUC | AP |
|---|---|---:|---:|---:|---:|---:|
| U-Net (A) | CV G1 (fold 1-4) | 73,3 | 74,1 | 57,9 | 0,977 | 0,805 |
| U-Net (A) | Fold 5 (G2-G5) | 21,9 | 49,0 | 17,8 | 0,927 | 0,228 |
| U-Net (A) | Validare | 47,9 | 71,6 | 40,2 | 0,942 | 0,657 |
| U-Net (A) | Test oficial | 48,5 | 76,9 | 42,3 | 0,967 | 0,666 |
| RF pe pixeli (A) | CV G1 (fold 1-4) | 64,8 | 72,2 | 51,8 | 0,988 | 0,746 |
| RF pe pixeli (A) | Fold 5 (G2-G5) | 8,7 | 16,2 | 6,0 | 0,863 | 0,110 |
| RF pe pixeli (A) | Validare | 41,7 | 48,6 | 28,9 | 0,959 | 0,444 |
| RF pe pixeli (A) | Test oficial | 42,1 | 49,2 | 29,3 | 0,963 | 0,433 |
| Prag NDVI (A) | CV G1 (fold 1-4) | 18,0 | 64,8 | 16,3 | 0,913 | 0,165 |
| Prag NDVI (A) | Fold 5 (G2-G5) | 9,3 | 80,7 | 9,1 | 0,925 | 0,104 |
| Prag NDVI (A) | Validare | 22,3 | 79,6 | 21,1 | 0,948 | 0,297 |
| Prag NDVI (A) | Test oficial | 22,4 | 75,6 | 20,9 | 0,941 | 0,249 |
| U-Net (B) | CV G1 (fold 1-4) | 4,1 | 33,7 | 3,7 | 0,651 | 0,041 |
| U-Net (B) | Fold 5 (G2-G5) | 2,7 | 13,6 | 2,2 | 0,629 | 0,021 |
| U-Net (B) | Validare | 2,6 | 15,4 | 2,3 | 0,598 | 0,025 |
| U-Net (B) | Test oficial | 3,1 | 16,7 | 2,7 | 0,616 | 0,028 |
| RF pe pixeli (B) | CV G1 (fold 1-4) | 3,4 | 14,7 | 2,8 | 0,589 | 0,033 |
| RF pe pixeli (B) | Fold 5 (G2-G5) | 0,9 | 4,2 | 0,8 | 0,332 | 0,010 |
| RF pe pixeli (B) | Validare | 2,1 | 9,7 | 1,7 | 0,561 | 0,020 |
| RF pe pixeli (B) | Test oficial | 2,4 | 10,6 | 2,0 | 0,565 | 0,023 |
| Prag pantă (B) | CV G1 (fold 1-4) | 4,0 | 31,8 | 3,6 | 0,637 | 0,041 |
| Prag pantă (B) | Fold 5 (G2-G5) | 1,8 | 6,8 | 1,4 | 0,685 | 0,023 |
| Prag pantă (B) | Validare | 2,7 | 10,9 | 2,2 | 0,567 | 0,024 |
| Prag pantă (B) | Test oficial | 3,4 | 13,5 | 2,8 | 0,581 | 0,030 |

## Comparații pereche: ΔF1 (puncte) [IC 95%], p = fracția reeșantionărilor cu Δ ≤ 0

| Comparație | CV G1 (fold 1-4) | Fold 5 (G2-G5) | Validare | Test oficial |
|---|---:|---:|---:|---:|
| U-Net (A) − RF pe pixeli (A) | +5,1 [+4,7, +5,6], p=0,000 | +18,9 [+8,9, +27,2], p=0,000 | +12,5 [+9,4, +15,9], p=0,000 | +14,1 [+12,0, +16,2], p=0,000 |
| U-Net (A) − Prag NDVI (A) | +45,6 [+43,8, +47,2], p=0,000 | +13,6 [+7,3, +18,8], p=0,000 | +22,5 [+5,5, +35,0], p=0,003 | +24,9 [+19,5, +29,7], p=0,000 |
| U-Net (B) − RF pe pixeli (B) | +1,8 [+1,3, +2,2], p=0,000 | +2,8 [+1,7, +3,7], p=0,000 | +1,0 [-0,1, +2,4], p=0,045 | +1,2 [+0,4, +2,1], p=0,000 |
| U-Net (B) − Prag pantă (B) | +0,3 [-0,1, +0,6], p=0,062 | +1,5 [+0,7, +2,2], p=0,000 | +0,2 [-1,0, +1,5], p=0,376 | -0,2 [-1,2, +0,8], p=0,646 |
| U-Net (A) − U-Net (B) | +66,1 [+64,7, +67,2], p=0,000 | +25,9 [+16,8, +33,4], p=0,000 | +52,9 [+41,0, +61,9], p=0,000 | +54,2 [+51,3, +57,5], p=0,000 |

## Metrica pentru decizie: captura alunecărilor verificând primele 10% / 20% patch-uri (%)

Patch-urile (~1,28 km) se ordonează după scorul mediu prezis; captura = fracția pixelilor de alunecare din patch-urile verificate. Ordonarea la întâmplare dă în medie 10%, respectiv 20%. AUC patch = separarea patch-urilor cu / fără alunecare.

| Model | Evaluare | Captura top 10% | Captura top 20% [IC 95%] | AUC patch |
|---|---|---:|---:|---:|
| U-Net (A) | CV G1 (fold 1-4) | 51,4 | 73,2 [70,5, 75,0] | 0,953 |
| U-Net (A) | Fold 5 (G2-G5) | 37,2 | 73,1 [54,2, 86,1] | 0,897 |
| U-Net (A) | Validare | 45,5 | 62,7 [47,3, 75,0] | 0,790 |
| U-Net (A) | Test oficial | 55,7 | 72,5 [66,6, 77,1] | 0,825 |
| RF pe pixeli (A) | CV G1 (fold 1-4) | 46,1 | 66,4 [63,1, 68,9] | 0,822 |
| RF pe pixeli (A) | Fold 5 (G2-G5) | 13,5 | 32,8 [20,6, 57,5] | 0,819 |
| RF pe pixeli (A) | Validare | 42,1 | 49,1 [19,4, 65,8] | 0,437 |
| RF pe pixeli (A) | Test oficial | 34,3 | 49,1 [38,9, 58,0] | 0,514 |
| Prag NDVI (A) | CV G1 (fold 1-4) | 17,2 | 36,8 [32,7, 40,3] | 0,602 |
| Prag NDVI (A) | Fold 5 (G2-G5) | 40,2 | 62,5 [43,9, 75,1] | 0,759 |
| Prag NDVI (A) | Validare | 38,7 | 43,0 [8,5, 62,6] | 0,294 |
| Prag NDVI (A) | Test oficial | 19,8 | 35,9 [22,8, 45,0] | 0,325 |
| U-Net (B) | CV G1 (fold 1-4) | 17,2 | 31,8 [28,7, 34,8] | 0,719 |
| U-Net (B) | Fold 5 (G2-G5) | 18,2 | 31,2 [23,2, 39,9] | 0,767 |
| U-Net (B) | Validare | 10,5 | 26,9 [13,7, 45,5] | 0,618 |
| U-Net (B) | Test oficial | 15,2 | 30,3 [24,9, 39,0] | 0,635 |
| RF pe pixeli (B) | CV G1 (fold 1-4) | 16,8 | 31,2 [27,8, 33,7] | 0,709 |
| RF pe pixeli (B) | Fold 5 (G2-G5) | 8,1 | 10,9 [6,2, 22,1] | 0,449 |
| RF pe pixeli (B) | Validare | 6,7 | 27,1 [12,8, 45,6] | 0,615 |
| RF pe pixeli (B) | Test oficial | 13,5 | 27,4 [20,8, 35,6] | 0,650 |
| Prag pantă (B) | CV G1 (fold 1-4) | 16,5 | 29,9 [26,7, 32,9] | 0,734 |
| Prag pantă (B) | Fold 5 (G2-G5) | 8,1 | 19,6 [12,4, 33,2] | 0,747 |
| Prag pantă (B) | Validare | 8,2 | 29,8 [11,8, 51,1] | 0,639 |
| Prag pantă (B) | Test oficial | 15,8 | 28,9 [21,3, 37,3] | 0,665 |

ΔCaptura top 20% (puncte) [IC 95%]:

- U-Net (A) − RF pe pixeli (A): CV G1 (fold 1-4) +6,7 [+5,0, +8,5]; Fold 5 (G2-G5) +40,3 [+24,6, +49,9]; Validare +13,6 [-8,4, +39,8]; Test oficial +23,3 [+17,9, +29,9]
- U-Net (A) − Prag NDVI (A): CV G1 (fold 1-4) +36,3 [+33,0, +39,5]; Fold 5 (G2-G5) +10,6 [-4,2, +33,9]; Validare +19,7 [-4,5, +47,6]; Test oficial +36,6 [+30,2, +47,0]
- U-Net (B) − RF pe pixeli (B): CV G1 (fold 1-4) +0,7 [-1,7, +3,2]; Fold 5 (G2-G5) +20,3 [+6,9, +29,5]; Validare -0,2 [-8,3, +17,1]; Test oficial +2,9 [-0,9, +10,5]
- U-Net (B) − Prag pantă (B): CV G1 (fold 1-4) +2,0 [-1,1, +4,8]; Fold 5 (G2-G5) +11,6 [+3,1, +16,7]; Validare -2,9 [-11,1, +9,7]; Test oficial +1,4 [-2,9, +8,9]
- U-Net (A) − U-Net (B): CV G1 (fold 1-4) +41,3 [+37,2, +44,8]; Fold 5 (G2-G5) +42,0 [+27,8, +53,5]; Validare +35,8 [+18,6, +51,4]; Test oficial +42,1 [+31,3, +49,2]

## Epoca aleasă la U-Net (validarea internă)

- U-Net (A): epoca 5: 1, epoca 10: 1, epoca 11: 3, epoca 12: 4, epoca 13: 3, epoca 14: 4, epoca 15: 2 (din 18 rulări; maximul e 15)
- U-Net (B): epoca 5: 2, epoca 6: 2, epoca 7: 1, epoca 8: 2, epoca 9: 1, epoca 11: 3, epoca 12: 3, epoca 13: 3, epoca 15: 1 (din 18 rulări; maximul e 15)
