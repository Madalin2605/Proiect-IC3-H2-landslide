# Tabel SOA: detecția și susceptibilitatea alunecărilor (H2)

| # | Articol | Metodă | Date | Metrică | Rezultat |
|---|---|---|---|---|---|
| 1 | Ghorbanzadeh et al., 2019 | ANN, SVM, RF și CNN-uri | RapidEye și factori topografici, districtul Rasuwa (Nepal); zone separate de antrenare și test | mIoU | 78,26%, cel mai bun rezultat, cu CNN pe fereastră mică |
| 2 | Sameen & Pradhan, 2019 | CNN și ResNet, cu fuziune prin stivuire sau pe trăsături | Fotografii aeriene RGB și topografie din LiDAR, Cameron Highlands (Malaezia); situri separate | F1, mIoU | ResNet cu fuziune pe trăsături: F1 0,90 și mIoU 90,24%. Doar topografie, CNN: F1 0,67 |
| 3 | Ghorbanzadeh et al., 2022a | 11 rețele de segmentare, cea mai bună fiind ResU-Net | Landslide4Sense, 3.799 de patch-uri din 4 situri; un sfert din fiecare zonă la antrenare, restul la test | F1 | ResU-Net 71,65%; U-Net 69,94% |
| 4 | Ghorbanzadeh et al., 2022b | Câștigător: ansamblu Swin, EfficientNetV2 și SegFormer, cu pseudo-etichete | Landslide4Sense, test oficial de 800 de patch-uri din alte situri | F1 pe clasa alunecare | 74,54%; baseline U-Net 59,92% |
| 5 | Le et al., 2023 | U-Net cu blocuri reziduale, atenție și mai multe capete de ieșire | Landslide4Sense, cele 3.799 de patch-uri etichetate, validare încrucișată în 5 părți | F1, mIoU | 84,07% și 76,07% |
| 6 | Li et al., 2025 | Prithvi-EO-2.0-300M comparat cu CNN-uri, transformere și alte modele fundamentale | Landslide4Sense, test oficial, plus seturi externe | F1, mIoU | Prithvi 60,70% și 71,30%; U-Net 59,66% și 70,44% |
| 7 | Ado et al., 2022 | Sinteză a literaturii, protocol ROSES | Studii de susceptibilitate cu ML | AUC | Multe modele raportează AUC peste 0,90 |
| 8 | Merghadi et al., 2020 | Comparație și sinteză de algoritmi ML | Studii de susceptibilitate | Performanță comparată | Ansamblurile de arbori se descurcă bine; RF este robust cu reglaje puține (concluzie calitativă) |
| 9 | Steger et al., 2017 | Modele statistice și modele cu efecte mixte | Inventare incomplete de alunecări superficiale | Performanță predictivă și plauzibilitate | Inventarele foarte incomplete dau performanță aparent mare, dar cele mai eronate predicții (concluzie calitativă) |
| 10 | Sirbu, 2023 (rezumat de conferință) | Random Forest cu 1.501 arbori, split aleator 70/30 | 7.360 de unități de versant, 26 de atribute, centrul Italiei | AUC | 0,829 și 0,817 pentru cele două clasificări |

## Ce rezultă pentru proiect

- **Rezultatele nu sunt direct comparabile.** Pe Landslide4Sense, F1 variază între aproximativ 60% și 84% în funcție de protocol. Articolele 4 și 6 folosesc testul oficial din alte situri, iar articolul 5 folosește validare încrucișată pe setul de train, unde patch-urile vecine pot ajunge în părți diferite.
- **Baseline de referință pe testul oficial:** U-Net, cu F1 aproximativ 60%, iar cel mai bun rezultat din competiție este 74,54%.
- **Topografia singură are putere limitată, dar nenulă.** În articolul 2, F1 ajunge la 0,67 doar cu variabile topografice, pe alt set de date. Asta motivează comparația S2 + DEM/pantă cu doar DEM/pantă.
- **Fuziunea contează.** În articolul 2, fuziunea pe trăsături depășește stivuirea cu 0,13 F1 la ResNet.
- **Susceptibilitatea se evaluează de obicei cu AUC,** iar metrica ridicată nu garantează o hartă plauzibilă (articolul 9). De aici limita asumată a proxy-ului topografic.
- **Golul urmărit:** în articolele parcurse nu am găsit o comparație S2 + DEM/pantă față de doar DEM/pantă pe Landslide4Sense, cu split care evită vecinătatea patch-urilor.

## Referințe

1. Ghorbanzadeh, O., Blaschke, T., Gholamnia, K., Meena, S. R., Tiede, D., Aryal, J. (2019). Evaluation of Different Machine Learning Methods and Deep-Learning Convolutional Neural Networks for Landslide Detection. *Remote Sensing*, 11(2), 196. DOI: [10.3390/rs11020196](https://doi.org/10.3390/rs11020196)
2. Sameen, M. I., Pradhan, B. (2019). Landslide Detection Using Residual Networks and the Fusion of Spectral and Topographic Information. *IEEE Access*, 7, 114363–114373. DOI: [10.1109/ACCESS.2019.2935761](https://doi.org/10.1109/ACCESS.2019.2935761)
3. Ghorbanzadeh, O., Xu, Y., Ghamisi, P., Kopp, M., Kreil, D. (2022a). Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection. *IEEE Transactions on Geoscience and Remote Sensing*, 60, 1–17. DOI: [10.1109/TGRS.2022.3215209](https://doi.org/10.1109/TGRS.2022.3215209)
4. Ghorbanzadeh, O., Xu, Y., Zhao, H., et al. (2022b). The Outcome of the 2022 Landslide4Sense Competition: Advanced Landslide Detection From Multisource Satellite Imagery. *IEEE JSTARS*, 15, 9927–9942. DOI: [10.1109/JSTARS.2022.3220845](https://doi.org/10.1109/JSTARS.2022.3220845). Secțiunea de rezultate dă 74,54%, iar concluziile preprintului menționează 73,07%.
5. Le, C., Pham, L., Lampert, J., Schlögl, M., Schindler, A. (2023). Landslide Detection and Segmentation Using Remote Sensing Images and Deep Neural Network. arXiv: [2312.16717](https://arxiv.org/abs/2312.16717)
6. Li, W., Wang, S., Lee, H., Lu, C., Roy, S., Ramachandran, R., Hsu, C.-Y. (2025). Landslide Hazard Mapping with Geospatial Foundation Models: Geographical Generalizability, Data Scarcity, and Band Adaptability. arXiv: [2511.04474](https://arxiv.org/abs/2511.04474)
7. Ado, M., Amitab, K., Maji, A. K., Jasińska, E., Gono, R., Leonowicz, Z., Jasiński, M. (2022). Landslide Susceptibility Mapping Using Machine Learning: A Literature Survey. *Remote Sensing*, 14(13), 3029. DOI: [10.3390/rs14133029](https://doi.org/10.3390/rs14133029)
8. Merghadi, A., Yunus, A. P., Dou, J., Whiteley, J., ThaiPham, B., Bui, D. T., Avtar, R., Abderrahmane, B. (2020). Machine learning methods for landslide susceptibility studies: A comparative overview of algorithm performance. *Earth-Science Reviews*, 207, 103225. DOI: [10.1016/j.earscirev.2020.103225](https://doi.org/10.1016/j.earscirev.2020.103225)
9. Steger, S., Brenning, A., Bell, R., Glade, T. (2017). The influence of systematically incomplete shallow landslide inventories on statistical susceptibility models and suggestions for improvements. *Landslides*, 14(5), 1767–1781. DOI: [10.1007/s10346-017-0820-0](https://doi.org/10.1007/s10346-017-0820-0)
10. Sirbu, F. (2023). Landslide Susceptibility Model based on Random Forest classification. EGU General Assembly 2023, EGU23-733. DOI: [10.5194/egusphere-egu23-733](https://doi.org/10.5194/egusphere-egu23-733)
