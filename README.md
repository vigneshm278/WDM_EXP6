### EX6 Information Retrieval Using Vector Space Model in Python
### NAME: VIGNESH M
### DATE: 04.09.2025
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```
    import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

# --- Build TF-IDF ---
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents.values()).toarray()
terms = vectorizer.get_feature_names_out()
idf_values = vectorizer.idf_

# --- Term Frequency (TF) ---
tf_matrix = np.zeros_like(tfidf_matrix, dtype=int)
for i, doc in enumerate(documents.values()):
    words = doc.lower().split()
    for j, term in enumerate(terms):
        tf_matrix[i, j] = words.count(term)

# --- Document Frequency (DF) ---
df = np.sum(tf_matrix > 0, axis=0)

# --- Magnitude of each document vector ---
magnitudes = np.linalg.norm(tfidf_matrix, axis=1)

# --- Cosine similarity and dot product ---
def cosine_and_dot(query_vec, doc_vec):
    dot = np.dot(query_vec, doc_vec)
    norm_query = np.linalg.norm(query_vec)
    norm_doc = np.linalg.norm(doc_vec)
    if norm_query == 0 or norm_doc == 0:
        cosine = 0.0  # instead of NaN
    else:
        cosine = dot / (norm_query * norm_doc)
    return cosine, dot


# --- Get query from user ---
query = input("Enter your query: ")
query_vec = vectorizer.transform([query]).toarray()[0]

# --- Build full table ---
rows = []
for i, doc_id in enumerate(documents.keys()):
    cosine, dot = cosine_and_dot(query_vec, tfidf_matrix[i])
    for j, term in enumerate(terms):
        rows.append({
            "Document": doc_id,
            "Term": term,
            "TF": tf_matrix[i, j],
            "DF": int(df[j]),
            "IDF": round(idf_values[j], 4),
            "Weight(TF*IDF)": round(tfidf_matrix[i, j], 4),
            "Magnitude": round(magnitudes[i], 4),
            "Cosine with Query": round(cosine, 4),
            "Dot with Query": round(dot, 4)
        })

df_table = pd.DataFrame(rows)
print("\n=== Full TF-IDF & Similarity Table ===")
print(df_table)
```
### Output:
<img width="880" height="485" alt="image" src="https://github.com/user-attachments/assets/577e56e2-2d9f-4ea5-8d41-85cadcf28c3d" />



### Result:
Thus the code to implement Information Retrieval Using Vector Space Model in Python is executed successfully
