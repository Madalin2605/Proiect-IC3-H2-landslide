# Data card: Landslide4Sense (tema H2)

Cifrele marcate cu **[măsurat]** sunt calculate din fișierele locale de `scripts/check_data.py`, `scripts/check_structure.py` și `scripts/check_mask_match.py`. Restul provine din documentația setului.

## Rezumat

| Câmp | Valoare |
|---|---|
| Set | Landslide4Sense 2022 |
| Rol în proiect | Detecția alunecărilor produse (țintă principală) și proxy topografic de susceptibilitate |
| Sursă originală | [github.com/iarai/landslide4sense-2022](https://github.com/iarai/landslide4sense-2022) (IARAI) |
| Copia folosită | Imagini și măști de train: [Kaggle, tekbahadurkshetri/landslide4sense](https://www.kaggle.com/datasets/tekbahadurkshetri/landslide4sense), versiunea 1. Măști de validare și test: [Hugging Face, ibm-nasa-geospatial/Landslide4sense](https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense) |
| Referință | Ghorbanzadeh et al., *IEEE JSTARS*, 15, 9927–9942, 2022, DOI [10.1109/JSTARS.2022.3220845](https://doi.org/10.1109/JSTARS.2022.3220845) |
| Perioadă | 2015–2021 conform README-ului IARAI, fără dată de achiziție pe patch. Articolul de benchmark menționează evenimente din 2009, 2015 și 2018 |
| Acoperire | Train: 4 situri (Iburi-Tobu, Japonia; Kodagu, India; Rasuwa, Nepal; vestul Taitung, Taiwan). Validare și test: alte situri, nenumite. Coordonatele pe patch nu sunt publicate |
| Rezoluție spațială | ~10 m/pixel, toate benzile fiind redimensionate la această rezoluție |
| Rezoluție temporală | Câte o observație pe patch, fără serie de timp |
| Eșantioane | 4.844 de patch-uri de 128×128 pixeli **[măsurat]** |
| Format | HDF5: imagine `float64` 128×128×14 (`img`), mască `uint8` 128×128 (`mask`) **[măsurat]** |
| Volum local | 8,4 GB **[măsurat]** |
| Licență | Pagina Kaggle indică „Other” și „Licensed by IARAI GmbH”, fără termeni detaliați. MIT acoperă doar codul din repo-ul IARAI; GitHub și Hugging Face nu declară licența datelor |

## Surse și benzi

Sursele sunt deja aliniate pe același patch de către autorii setului.

| Index în tablou | Nume | Sursă | Observație |
|---|---|---|---|
| 0–11 | B1–B12 | Sentinel-2 | 12 benzi multispectrale, fără banda 8A |
| 12 | Pantă (B13) | ALOS PALSAR | Derivată din relief |
| 13 | DEM (B14) | ALOS PALSAR | Altitudine |
| mască | 0 / 1 | Etichete ale setului | 1 = alunecare, 0 = restul |

Documentația indică ALOS PALSAR pentru DEM și pantă. Le et al. (2023) observă că produsele ALOS PALSAR RTC folosesc de fapt DEM-uri externe (SRTM și NED), deci originea exactă a reliefului rămâne neclară.

**CHIRPS nu este folosit în primul pipeline.** Fără coordonate și date de achiziție nu poate fi aliniat cu patch-urile.

## Unități și valori

Documentația nu precizează unitățile sau scalarea. Valorile nu sunt reflectanțe, metri sau grade în forma din fișier **[măsurat]**:

- Benzile Sentinel-2 au media pe train între 0,92 și 1,17, deci par scalate cu o constantă pe bandă. Aceasta este o deducție, nu o informație documentată.
- Panta are valori între 0 și 4,04, iar DEM între 0 și 5,12 pe train.
- Statisticile complete pe bandă și split sunt în [`reports/band_stats.csv`](../reports/band_stats.csv).

## Împărțire train / validare / test

| Split | Patch-uri | Măști | Patch-uri cu alunecări | Pixeli alunecare |
|---|---:|---:|---:|---:|
| Train | 3.799 | 3.799 | 2.231 (58,7%) | 1.442.790 din 62.242.816 (2,32%) |
| Validation | 245 | 245 | 146 (59,6%) | 68.745 din 4.014.080 (1,71%) |
| Test | 800 | 800 | 536 (67,0%) | 247.531 din 13.107.200 (1,89%) |

Împărțirea este cea oficială a competiției. README-ul IARAI spune că măștile există doar pentru train, așa că măștile de validare și test provin de pe Hugging Face, iar imaginile de pe Kaggle.

**Combinarea celor două surse a fost verificată [măsurat].** NDVI este mai mic în interiorul măștii decât în afara ei la 100% din patch-urile de validare cu alunecări și la 99,8% din cele de test. La perechi imagine–mască amestecate, proporția scade la 47% și 46%. Rezultatul este în [`reports/mask_match.json`](../reports/mask_match.json).

## Controale de calitate [măsurat]

| Control | Rezultat |
|---|---|
| Formă imagini și măști | Toate corecte |
| NaN sau infinit | 0 patch-uri |
| Corespondență imagine–mască, pe ID | Completă în toate split-urile |
| Valori mască | Doar 0 și 1 |
| Imagini duplicate exact, în toate split-urile | 0 |

## Goluri

- **Relief constant:** 115 patch-uri de train au DEM constant (0,031 sau 0,032), iar 114 au și panta constantă. Niciunul nu conține alunecări. Cauza nu este documentată; apa sau zonele plate sunt doar o ipoteză.
- **Pixeli zero în toate benzile:** 7.423 în 57 de patch-uri de train, 256 în 2 patch-uri de validare și 512 în 4 patch-uri de test.

## Leakage și limite

- **ID-urile de train consecutive sunt patch-uri vecine.** La 97,9% din cele 3.666 de perechi consecutive, marginea DEM continuă mai bine decât pragul de 5% al perechilor aleatoare (diferență mediană 0,005, față de 1,014). Un split aleator în train ar pune zone lipite în ambele părți, deci umflă rezultatele.
- La validare și test, ID-urile consecutive nu arată această continuitate (7,0% și 5,9%, apropiat de nivelul aleator de 5%).
- Validarea și testul oficial provin din alte situri decât train-ul, deci sunt un test geografic real. Split-ul spațial în interiorul train-ului nu se poate face din metadate; la CP2 se folosesc blocuri compacte de ID-uri.
- Distribuțiile diferă între split-uri: de exemplu, media benzii B4 este 0,96 pe train și 1,21 pe validare, iar abaterea DEM este 1,07 pe train și 0,76 pe validare.
- Dezechilibru mare: prezicerea „fără alunecare” peste tot dă acuratețe 97,68% pe train, 98,29% pe validare și 98,11% pe test, dar F1 = 0, deci acuratețea nu este o metrică potrivită.
- Măștile de validare și test au fost folosite în competiție ca etichete ascunse. Le folosim doar pentru evaluare, nu la antrenare sau la alegerea hiperparametrilor pe test.
- Patch-urile provin doar din zone afectate. Pixelii neetichetați nu sunt garantat stabili, iar setul nu conține litologie. Proxy-ul topografic nu este o hartă reală de susceptibilitate.
- Rezultatele nu se transferă automat la un județ din România; este nevoie de validare locală.
