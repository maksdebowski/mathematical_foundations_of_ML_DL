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