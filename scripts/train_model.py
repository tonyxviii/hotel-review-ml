import pandas as pd 
import matplotlib.pyplot as plt
import joblib
import os 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# 1. CARICAMENTO DEI DATI

df = pd.read_csv("data/reviews.csv")

print("Dataset caricato.")
print(f"Totale recensioni: {len(df)}")
print(f"Colonne: {list(df.columns)}")

# Combiniamo titolo e testo in un unico campo
df["full_text"] = df["title"] + " " + df["body"]

# 2. PREPARAZIONE DEI DATI

x = df["full_text"]           # Input: il testo della recensione
y_dept = df["department"]     # Output 1: il reparto
y_sent = df["sentiment"]      # Output 2: il sentiment

# SPlit 80% training, 20% test
X_train, X_test, y_dept_train, y_dept_test = train_test_split(
    x, y_dept, test_size=0.2, random_state=42
)

_, _, y_sent_train, y_sent_test = train_test_split(
    x, y_sent, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train)} recensioni")
print(f"Test set: {len(X_test)} recensioni")

# 3. CREAZIONE DELLE PIPELINE ML

# TfidfVectorizer: traforma il testo in numeri
# LogisticRegression: il modello che impara dai numeri

pipeline_dept = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

pipeline_sent = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# 4. ADDESTRAMENTO

print("\nAddestramento modello REPARTO in corso...")
pipeline_dept.fit(X_train, y_dept_train)
print("Modello reaparto addestrato.")

print("Addestramento modello SENTIMENT in corso...")
pipeline_sent.fit(X_train, y_sent_train)
print("Modello sentiment addestrato.")

# 5. VALUTAZIONE

# Predizioni sul test set 
y_dept_pred = pipeline_dept.predict(X_test)
y_sent_pred = pipeline_sent.predict(X_test)

# Calcolo metriche 
dept_accuracy = accuracy_score(y_dept_test, y_dept_pred)
dept_f1 = f1_score(y_dept_test, y_dept_pred, average="macro")

sent_accuracy = accuracy_score(y_sent_test, y_sent_pred)
sent_f1 = f1_score(y_sent_test, y_sent_pred, average="macro")

print("\n========== RISULTATI ==========")
print(f"REPARTO - Accuracy: {dept_accuracy: .2%} | F1 Macro: {dept_f1: .2%}")
print(f"SENTIMENT - Accuracy: {sent_accuracy: .2%} | F1 Macro: {sent_f1: .2%}")
print("================================")

# 6. CONFUSION MATRIX

# Cofusion matrix per il reparto 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_dept = confusion_matrix(y_dept_test, y_dept_pred, labels=["Housekeeping", "Reception", "F&B"])
disp_dept = ConfusionMatrixDisplay(confusion_matrix=cm_dept, display_labels=["Housekeeping", "Reception", "F&B"])
disp_dept.plot(ax=axes[0], colorbar = False)
axes[0].set_title("Confusion Matrix - Reparto")

# Confusion matrix per il sentiment 
cm_sent = confusion_matrix(y_sent_test, y_sent_pred, labels=["positive", "negative"])
disp_sent = ConfusionMatrixDisplay(confusion_matrix=cm_sent, display_labels=["positive", "negative"])
disp_sent.plot(ax=axes[1], colorbar = False)
axes[1].set_title("Confusion Matrix - Sentiment")

plt.tight_layout()
plt.savefig("data/confusion_matrices.png", dpi=300)
plt.show()
print("\nGrafici salvati in data/confusion_matrices.png")


# 6B. BAR CHART PER CLASSE

# Contiamo le predizioni per reparto
dept_counts = pd.Series(y_dept_pred).value_counts()
sent_counts = pd.Series(y_sent_pred).value_counts()

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart reparto
axes2[0].bar(dept_counts.index, dept_counts.values,
             color=["#4CAF50", "#2196F3", "#FF9800"])
axes2[0].set_title("Predizioni per Reparto")
axes2[0].set_xlabel("Reparto")
axes2[0].set_ylabel("Numero di recensioni")
for i, v in enumerate(dept_counts.values):
    axes2[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

# Bar chart sentiment
axes2[1].bar(sent_counts.index, sent_counts.values,
             color=["#4CAF50", "#F44336"])
axes2[1].set_title("Predizioni per Sentiment")
axes2[1].set_xlabel("Sentiment")
axes2[1].set_ylabel("Numero di recensioni")
for i, v in enumerate(sent_counts.values):
    axes2[1].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("data/bar_charts.png", dpi=150)
plt.show()
print("Bar chart salvato in data/bar_charts.png")

# 7. SALVATAGGIO DEI MODELLI

os.makedirs("models", exist_ok = True)
joblib.dump(pipeline_dept, "models/model_department.pkl")
joblib.dump(pipeline_sent, "models/model_sentiment.pkl")
print("Modelli salvati nella cartella models/")

# 8. PREDIZIONI SU TUTTO IL DATASET (export CSV)

df["pred_department"] = pipeline_dept.predict(df["full_text"])
df["pred_sentiment"] = pipeline_sent.predict(df["full_text"])
df.to_csv("data/reviews_with_predictions.csv", index = False)
print("CSV con predizioni salvato in data/reviews_with_predictions.csv")

# 9. ESEMPI DI ERRORI TIPICI

print("\n========== ERRORI TIPICI ==========")

# Trovaimo le recensioni dove il modello ha sbaglaito il reparto 
errors_dept = X_test[y_dept_pred != y_dept_test].copy()
true_dept = y_dept_test[y_dept_pred != y_dept_test]
pred_dept = pd.Series(y_dept_pred, index = y_dept_test.index)[y_dept_pred != y_dept_test]

if len(errors_dept) == 0:
    print("Nessun errore nel reparto - modello perfetto sul test set.")
else: 
    print(f"\nErrori classificazione REPARTO ({len(errors_dept)} errori):")
    for i, (text, true, pred) in enumerate(zip(errors_dept, true_dept, pred_dept)):
        print(f"\nEsempio {i+1}:")
        print(f" Testo: {text[:80]}...")
        print(f" Reparto reale:     {true}")
        print(f" Reparto predetto:  {pred}")

# Troviamo le recensioni dove il modello ha sbagliato il sentiment
errors_sent = X_test[y_sent_pred != y_sent_test].copy()
true_sent = y_sent_test[y_sent_pred != y_sent_test]
pred_sent = pd.Series(y_sent_pred, index=y_sent_test.index)[y_sent_pred != y_sent_test]

if len(errors_sent) == 0:
    print("\nNessun errore nel sentiment — modello perfetto sul test set.")
else:
    print(f"\nErrori classificazione SENTIMENT ({len(errors_sent)} errori):")
    for i, (text, true, pred) in enumerate(zip(errors_sent, true_sent, pred_sent)):
        print(f"\nEsempio {i+1}:")
        print(f"  Testo: {text[:80]}...")
        print(f"  Sentiment reale:    {true}")
        print(f"  Sentiment predetto: {pred}")

print("\n====================================")
