# Product Title Classification

Ovaj projekat implementira **automatsku klasifikaciju naslova proizvoda** u kategorije koristeći modele mašinskog učenja i TF-IDF vektorizaciju. 
Projekat uključuje trening modela, evaluaciju performansi i interaktivno predviđanje kategorija za nove proizvode.

## Struktura projekta

```
Vesna_Sibinovic_Task3/
│
├─ data/
│   └─ products.csv              # Ulazni CSV fajl sa podacima
│
├─ src/
│   ├─ train_model.py            # Skripta za trening i evaluaciju modela
│   ├─ predict_category.py       # Skripta za predviđanje kategorija novih proizvoda
│   ├─ final_model.pkl           # Sačuvani najbolji model (SVM)
│   └─ vectorizer.pkl            # Sačuvani TF-IDF vektorizator
│
├─ notebooks/
│   └─ Product_Classification.ipynb  # Jupyter notebook sa treningom i vizualizacijom
│
├─ README.md
└─ requirements.txt
```

---

## 1. Instalacija i priprema okruženja

1. Preporučuje se kreiranje virtualnog okruženja:

```bash
python -m venv venv
```

2. Aktivacija okruženja:

- Windows:
```bash
venv\Scripts\activate
```
- Linux / macOS:
```bash
source venv/bin/activate
```

3. Instalacija potrebnih paketa:

```bash
pip install -r requirements.txt
```

---

## 2. Trening modela

Trening se vrši pokretanjem `train_model.py` koji:

- Čisti tekst iz kolone `Product Title`
- Deluje TF-IDF vektorizator
- Trening tri modela:
  - Logistic Regression
  - Naive Bayes
  - SVM
- Prikazuje metrike performansi: accuracy, precision, recall, F1-score
- Vizualizuje tačnost modela
- Čuva najbolji model i TF-IDF vektorizator u `src/`

Pokretanje:

```bash
python src/train_model.py
```

Na kraju treninga, najbolji model (SVM) i TF-IDF vektorizator su sačuvani u folderu `src/`.

---

## 3. Predviđanje kategorija novih proizvoda

Skripta `predict_category.py` omogućava predviđanje kategorija za nove proizvode:

- Učitava sačuvani model i TF-IDF vektorizator
- Omogućava unos jednog ili više naslova proizvoda
- Vraća tabelu sa predviđenim kategorijama
- Opcionalno može sačuvati rezultate u CSV fajl

Pokretanje:

```bash
python src/predict_category.py
```

Primer unosa:

```
Unesite naslov proizvoda (ili pritisnite Enter za kraj): Samsung Galaxy S23
Unesite naslov proizvoda (ili pritisnite Enter za kraj):
Rezultati predikcije:
             Product Predicted Category
0  Samsung Galaxy S23      Mobile Phones
```

---

## 4. Zaključci

- Najbolji model za klasifikaciju naslova proizvoda je **SVM**.
- Postignuta tačnost na test skupu je ~0.95.
- Model i TF-IDF vektorizator su sačuvani i spremni za predviđanje novih proizvoda.
- Vizualizacija tačnosti modela omogućava brzu identifikaciju najboljeg klasifikatora.

---

## 5. Biblioteke

Projekat koristi sledeće biblioteke (fiksirane verzije u `requirements.txt`):

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- re (standardna Python biblioteka)

---

## 6. Napomene

- Svi `.pkl` i `.py` fajlovi se nalaze u folderu `src/`.
- CSV fajl sa podacima se nalazi u folderu `data/`.
- Jupyter notebook se nalazi u folderu `notebooks/` za vizuelnu analizu i eksperimentisanje.

