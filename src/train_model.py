import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Učitavanje podataka
# ------------------------------------------------------------
DATA_PATH = r"D:\It-akademija\Introduction to Machine Learning using Python\Tasks\Vesna_Sibinovic_Task3\data\products.csv"
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

TITLE_COL = "Product Title"
CATEGORY_COL = "Category Label"

# ------------------------------------------------------------
# 2. Čišćenje teksta
# ------------------------------------------------------------
def ocisti_tekst(tekst):
    tekst = re.sub(r'[^a-zA-Z\s]', '', str(tekst))
    tekst = tekst.lower()
    tekst = re.sub(r'\s+', ' ', tekst).strip()
    return tekst

data['clean_title'] = data[TITLE_COL].apply(ocisti_tekst)
data = data.dropna(subset=['clean_title', CATEGORY_COL])

# ------------------------------------------------------------
# 3. Deljenje podataka
# ------------------------------------------------------------
X = data['clean_title']
y = data[CATEGORY_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 4. TF-IDF vektorizacija
# ------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ------------------------------------------------------------
# 5. Trening modela i evaluacija
# ------------------------------------------------------------
modeli = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(random_state=42, class_weight="balanced")
}

rezultati = {}

for naziv, model in modeli.items():
    print(f"\n🔹 Treniram model: {naziv}")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Tačnost: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    rezultati[naziv] = {"model": model, "accuracy": acc}

# ------------------------------------------------------------
# 6. Vizualizacija rezultata
# ------------------------------------------------------------
tabela_rezultata = pd.DataFrame({
    "Model": list(rezultati.keys()),
    "Tačnost": [rezultati[m]["accuracy"] for m in rezultati]
}).sort_values(by="Tačnost", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="Tačnost", data=tabela_rezultata, palette="pastel", dodge=False)
plt.title("Uporedni prikaz tačnosti modela", fontsize=14)
plt.xlabel("Model")
plt.ylabel("Tačnost")
plt.ylim(0, 1)
plt.show()

# ------------------------------------------------------------
# 7. Odabir i čuvanje najboljeg modela
# ------------------------------------------------------------
najbolji_model_naziv = tabela_rezultata.iloc[0]['Model']
najbolji_model = rezultati[najbolji_model_naziv]["model"]
print(f"\nNajbolji model: {najbolji_model_naziv} sa tačnošću {rezultati[najbolji_model_naziv]['accuracy']:.4f}")

MODEL_PATH = "src/final_model.pkl"
VECTORIZER_PATH = "src/vectorizer.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(najbolji_model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("Model i TF-IDF vektorizator su sačuvani u folderu 'src/'")