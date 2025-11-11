import os
import pickle
import re
import pandas as pd

# ------------------------------------------------------------
# Funkcija za čišćenje teksta
# ------------------------------------------------------------
def ocisti_tekst(tekst):
    tekst = re.sub(r'[^a-zA-Z\s]', '', str(tekst))  # samo slova i razmak
    tekst = tekst.lower()
    tekst = re.sub(r'\s+', ' ', tekst).strip()
    return tekst

# ------------------------------------------------------------
# Učitavanje sačuvanog modela i TF-IDF vektorizatora
# ------------------------------------------------------------

# Folder gde se nalazi skripta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Putanje do modela i vektorizatora
MODEL_PATH = os.path.join(BASE_DIR, "final_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ------------------------------------------------------------
# Funkcija za predviđanje kategorije jednog ili više proizvoda
# ------------------------------------------------------------
def predvidi_kategorije(naslovi_proizvoda):
    """
    Naslovi_proizvoda: list ili Series naslova proizvoda
    Vraća: DataFrame sa kolonama 'Product' i 'Predicted Category'
    """
    # Ako je unet samo string, pretvori u listu
    if isinstance(naslovi_proizvoda, str):
        naslovi_proizvoda = [naslovi_proizvoda]
    
    # Čišćenje tekstova
    cisti_naslovi = [ocisti_tekst(t) for t in naslovi_proizvoda]
    
    # Transformacija u TF-IDF vektore
    X_vec = vectorizer.transform(cisti_naslovi)
    
    # Predikcija kategorija
    predikcije = model.predict(X_vec)
    
    # Kreiranje DataFrame-a
    df_rezultata = pd.DataFrame({
        "Product": naslovi_proizvoda,
        "Predicted Category": predikcije
    })
    
    return df_rezultata

# ------------------------------------------------------------
# Primer upotrebe
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Predikcija kategorija za nove proizvode\n")
    
    # Primer: unos više proizvoda
    naslovi = []
    while True:
        naslov = input("Unesite naslov proizvoda (ili pritisnite Enter za kraj): ")
        if not naslov:
            break
        naslovi.append(naslov)
    
    if naslovi:
        rezultati = predvidi_kategorije(naslovi)
        print("\nRezultati predikcije:\n")
        print(rezultati)
        
        # Opcija za čuvanje u CSV
        sacuvaj = input("\nDa li želite da sačuvate rezultate u CSV? (da/ne): ").lower()
        if sacuvaj == "da":
            csv_putanja = "predikcija_rezultata.csv"
            rezultati.to_csv(csv_putanja, index=False)
            print(f"Rezultati su sačuvani u fajlu '{csv_putanja}'")
    else:
        print("Nisu uneti naslovi proizvoda.")