# Możliwe Tematy

## Computer Vision:

_CAŁE TO CIEKAWE_

- Medyczne dane syntetyczne (augmentacja danych w klasyfikacji obrazow medycznych)
- Wykrywanie anomalii lub wąski zbiór danych medycznych i diffusion model.
- Geolokacja, model do grania w geoguessr.
- Pomocnik dla osób niewidomych, ostrzeganie przed przeszkodami.
- Wykrywanie tornand i huraganów gdzie uderzy i jak mocno.
- System wspomagania diagnostyki medycznej z analiza obrazow.
  (np. Kaggle: `https://www.kaggle.com/competitions/isic-2024-challenge/overview`)

## Szeregi Czasowe:

_CAŁE TO CIEKAWE_

- Detekcja anomalii w syntetycznych danych finansowych z wykorzystaniem modeli generatywnych i probabilistycznej walidacji statystycznej.
- Estymacja ryzyka inwestycyjnego z wykorzystaniem sieci neuronowych i weryfikacja statystyczną -> Value at Risk -> Conditional Value at Risk.
- Wykrywanie anomalii w danych finansowych -> Autoencodery, GRU, Bayesowskie modele, detekcja outlierów, reinforcement learning.
- System wykrywania anomalii czasowych z samoadaptacją.

## Neural PDE/SDE:

- PDE -> PENN
- Estymacja parametrów stochastycznych równań różniczkowych w finansach przy użyciu sieci neuronowych (Neural SDE)

## Analiza dźwieku

_CAŁE TO CIEKAWE_

- Rozpoznawanie gatunków ptaków po dźwiękach (Kaggle: `https://www.kaggle.com/competitions/birdclef-2025/overview`)

## Bayesowskie sieci neuronowe:

- Modelowanie ryzyka finansowego z rzetelnymi przedziałami ufności.
- Wykrywanie anomalii z kwantyfikacją niepewności (przemysł: logi maszynowe)
- Efektywne (i efektowne!) strojenie hiperparametrów złożonych modeli.
- Modelowanie zmian parametrów klimatycznych.

## DL

- American Sign Language Recognition (Kaggle: `https://www.kaggle.com/competitions/asl-fingerspelling/overview`)
- Child Mind Institute - detakcja stanów snu (Kaggle: `https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview`)
- Systemy rekomendacyjne (netflix, spotify etc.)



Propozycje:
1) Detekcja anomalii + generowanie danych; przyklad medyczny(analogicznie finansowy) - Generator tworzy obrazy z roznymi patologiami, rzadkimi, lub trudnymi do wykrycia; Uczy sie identyfikowac obszary podejrzane, a system generuje coraz trudniejsze przypadki, by zwiekszyc czulosc "detektora", jako dataset: waski zbior jakis rzadkich guzow, zmian patologicznych lub wczesnych stadiow chorob.
2) Augmentacja danych dla poprawy klasyfikacji(opisane dla medycyny, ale np. przewidywanie zachowan rynku w finansach); rozwiazujemy problem z imbalanced dataset(dosc czeste w finansach/medycynie), a system uczy sie rozrozniac miedzy wieloma(jak duzo - ciezko na tym etapie okreslic) kategoriami chorób. System generuje warianty schorzeń o róznym nasileniu; Mogloby byc potencjalnie wykorzystywane jako systemy diagnostyczne, klasyfikacja chorob, ocena zaawanoswania schorzen

W obu przypadkach temat jest dosc dobrze opisany(latwo znalezc badania/modele na ten temat) wiec nasz model mozna by w jakis sposob porownywac z modelami dostepnymi na ten moment w sieci(nie neuronowej, zwyklym arxiv :D ).
Badania na ktorych mozna sie jakos oprzec/"poprawic":

FINANSE:
https://www.researchgate.net/publication/346491432_Deep_Learning_Based_Hybrid_Computational_Intelligence_Models_for_Options_Pricing - w 2020 dodany, jest duza szansa na poprawe wynikow, by pokonac ich model(ich zalozeniem bylo pokonac Blacka-Scholesa)
https://cs230.stanford.edu/projects_fall_2019/reports/26260984.pdf - stock market prediction
https://cs230.stanford.edu/projects_winter_2021/reports/70667451.pdf - option pricing using DL
https://www.researchgate.net/publication/377393910_Deep_Learning_in_Stock_Market_Forecasting_Comparative_Analysis_of_Neural_Network_Architectures_Across_NSE_and_NYSE - podejście podobne, ale "giełda" na której będzie to sprawdzane zbudowana przez nas.
https://www.researchgate.net/publication/350334084_Detection_of_Anomaly_Stock_Price_Based_on_Time_Series_Deep_Learning_Models - detekcja anomali, znów 2020 rok; mozna cos dodac?

MEDYCYNA:
https://www.mdpi.com/2313-433X/9/4/81 - wygląda ciekawie, ale schowane za paywallem ;(
https://link.springer.com/article/10.1007/s11042-023-14817-z - klasyczne podejście, moglibysmy zmienić problem(brain mri?), zbudować coś do niego(moze jakas augmentacja i porownanie wynikow z dostepnymi danymi)
https://www.researchgate.net/publication/369846612_Simulation_based_evaluation_framework_for_deep_learning_unsupervised_anomaly_detection_on_brain_FDG-PET - anomaly detection
https://www.mdpi.com/1424-8220/23/7/3440 - tutaj do ewentualnej ewaluacji naszych syntetycznych danych.
