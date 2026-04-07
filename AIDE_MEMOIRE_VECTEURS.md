# 📌 Aide-Mémoire: Vecteurs et Similarité Cosinus

## 🎓 Formules clés

### 1. Produit scalaire
$$\vec{u} \cdot \vec{v} = \sum_{i=0}^{n-1} u_i \times v_i$$

**Exemple avec 3 dimensions:**
$$\vec{u} \cdot \vec{v} = u_0 \times v_0 + u_1 \times v_1 + u_2 \times v_2$$

**Cas réel (384 dimensions):**
```
dot = u[0]×v[0] + u[1]×v[1] + ... + u[383]×v[383]
```

### 2. Norme L2 (longueur)
$$||\vec{u}|| = \sqrt{\sum_{i=0}^{n-1} u_i^2}$$

**Exemple:**
$$||\vec{u}|| = \sqrt{u_0^2 + u_1^2 + u_2^2}$$

### 3. Similarité cosinus
$$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{||\vec{u}|| \times ||\vec{v}||}$$

**Si vecteurs normalisés (||u|| = ||v|| = 1):**
$$\cos(\theta) = \vec{u} \cdot \vec{v}$$

---

## 🔢 Calcul étape par étape

### Hypothèse
```
Vecteur 1 (requête):  [-0.023,  0.145, -0.089]  (3 dims pour simplicité)
Vecteur 2 (chunk):    [-0.021,  0.142, -0.085]
```

### Étape 1: Calcul produit scalaire
```
u · v = (-0.023) × (-0.021) + (0.145) × (0.142) + (-0.089) × (-0.085)
      = 0.000483 + 0.02059 + 0.007565
      = 0.028638
```

### Étape 2: Calcul normes
```
||u|| = √((-0.023)² + (0.145)² + (-0.089)²)
      = √(0.000529 + 0.021025 + 0.007921)
      = √0.029475
      = 0.17168

||v|| = √((-0.021)² + (0.142)² + (-0.085)²)
      = √(0.000441 + 0.020164 + 0.007225)
      = √0.02783
      = 0.16682
```

### Étape 3: Calcul similarité
```
cos(θ) = 0.028638 / (0.17168 × 0.16682)
       = 0.028638 / 0.028639
       = 1.0000  (quasi identiques!)
```

**Interprétation:** Score très proch de 1.0 = excellent match! ✅

---

## 🧮 Cas réseau neural (normalisation)

Quand on utilise SentenceTransformer avec `normalize_embeddings=True`:

```python
encoder = SentenceTransformer("all-MiniLM-L6-v2")
vec = encoder.encode(["texte"], normalize_embeddings=True)[0]

# Résultat garanti:
np.linalg.norm(vec) ≈ 1.0  ✓

# Donc:
similarity = np.dot(vec_requete, vec_chunk)
# (pas besoin de diviser par les normes)
```

---

## 📐 Représentation géométrique

```
En 2D (simplification):

        │
        │    Vecteur A
        │   /
        │  /  )θ
    ────┼─────── (axis 0)
        │\
        │ \ Vecteur B
        │

cos(θ) proche de +1.0 → θ proche de 0° → vecteurs PARALLÈLES
cos(θ) proche de  0.0 → θ = 90° → vecteurs ORTHOGONAUX  
cos(θ) proche de -1.0 → θ proche de 180° → vecteurs OPPOSÉS
```

---

## 💾 Stockage FAISS

### Structure mémoire
```
Fichier: faiss.index (binaire)

[Header]
├─ Type: IndexFlatIP
├─ Dimension: 384
├─ Count: 2847
└─ Metric: IP

[Vectors (binary)]
├─ Vec[0]: [4 bytes] × 384 = 1536 bytes
├─ Vec[1]: [4 bytes] × 384 = 1536 bytes
├─ Vec[2]: [4 bytes] × 384 = 1536 bytes
├─ ...
└─ Vec[2846]: [4 bytes] × 384 = 1536 bytes

Total: 2847 × 1536 = 4,370,432 bytes ≈ 4.2 MB
```

### Chargement
```python
import faiss
index = faiss.read_index("index/faiss.index")
# Tout en RAM maintenant

# Recherche: super rapide
distances, indices = index.search(query_vec, k=10)
# Time: ~1-5 ms
```

---

## 🔍 Recherche FAISS

### Pseudo-code
```python
def faiss_search(query_vec, k=10):
    # Step 1: Calculer score pour TOUS les vecteurs
    scores = []
    for i in range(num_vectors):
        score = dot_product(query_vec, stored_vec[i])
        scores.append((i, score))
    
    # Step 2: Trier par score décroissant
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Step 3: Retourner top-k
    return scores[:k]

# Temps: O(n × d)
#   n = nombre de vecteurs (2847)
#   d = dimensions (384)
# = 2847 × 384 ≈ 1 million d'opérations
# = 1-5 ms sur CPU moderne
```

---

## 🎯 Mise en œuvre pratique

### 1. Créer des embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Attention: normalize_embeddings=True est important!
embeddings = model.encode(
    texts,
    normalize_embeddings=True  # os garantit ||v|| = 1.0
)

# Résultats:
# embeddings.shape = (N, 384)
# dtype = float32
# chaque embeddings[i] a une norme ≈ 1.0
```

### 2. Créer l'index FAISS
```python
import faiss
import numpy as np

# embeddings est une matrice (N, 384) en float32
dim = embeddings.shape[1]  # 384

# Créer l'index
index = faiss.IndexFlatIP(dim)

# Ajouter tous les vecteurs
index.add(embeddings.astype(np.float32))

# Sauvegarder
faiss.write_index(index, "faiss.index")
```

### 3. Rechercher
```python
# Encoder la requête
query = "Quels résultats?"
query_vec = model.encode([query], normalize_embeddings=True)

# Charger l'index
index = faiss.read_index("faiss.index")

# Rechercher
distances, indices = index.search(query_vec, k=10)

# distances[0] = scores pour les 10 meilleurs résultats
# indices[0] = indices des chunks dans index
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    print(f"Rank {i+1}: Chunk {idx}, Score {dist:.6f}")
```

---

## 📊 Interprétation des scores

```
Score cosinus    │ Interprétation           │ Action
─────────────────┼──────────────────────────┼─────────────
  > 0.80         │ ✅ Très bon match        │ Retourner
  0.70 - 0.80    │ ✅ Bon match             │ Retourner
  0.50 - 0.70    │ 👍 Acceptable            │ Considérer
  0.30 - 0.50    │ ⚠️ Moyen                 │ Peut aider
  0.00 - 0.30    │ ❌ Faible                │ Ignorer
  < 0.00         │ ❌ Contraire             │ Ignorer
```

---

## 🚀 Optimisations avancées

### Pour plus de vecteurs (100K+)
```
Problème: Recherche O(n) devient lent pour très grands n

Solution 1: IVF (Inverted File)
  index = faiss.IndexIVFFlat(quantizer, dim, nlist)
  # Divise l'espace en clusters
  # Temps: O(sqrt(n) × d) au lieu de O(n × d)

Solution 2: Product Quantization (PQ)
  index = faiss.IndexIVFPQ(...)
  # Compresse les vecteurs
  # Taille: 90% plus petit
  # Temps: 90% plus rapide
  # Trade-off: perte légère de qualité

Solution 3: HNSW (Hierarchical Navigable Small World)
  import hnswlib
  index = hnswlib.Index(space='cosine', dim=384)
  # Graph-based, très rapide
```

---

## 🔐 Validation

### Vérifier normalisation
```python
import numpy as np

for vec in embeddings:
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.01, f"Not normalized: {norm}"

print("✓ Tous les vecteurs sont normalisés")
```

### Vérifier FAISS
```python
# Après créer l'index
index = faiss.read_index("faiss.index")

print(f"✓ Type: {type(index)}")
print(f"✓ Dimension: {index.d}")
print(f"✓ Nombre vecteurs: {index.ntotal}")
print(f"✓ Index opérationnel")
```

### Test requête
```python
# Requête simple
query = "test"
query_vec = model.encode([query], normalize_embeddings=True)

distances, indices = index.search(query_vec, k=5)

print(f"✓ Recherche fonctionne")
print(f"  Top score: {distances [0][0]:.6f}")
print(f"  Meilleur chunk: {indices[0][0]}")
```

---

## 🎓 Concepts clés

| Terme | Définition | Importance |
|-------|-----------|-----------|
| **Embedding** | Représentation numérique d'un texte (384 dimensions) | 🔴 Critique |
| **Normalisation** | Mettre norme à 1.0 (important pour cosine) | 🔴 Critique |
| **Produit scalaire** | Σ(u[i] × v[i]) | 🔴 Core du calcul |
| **Similarité cosinus** | (u·v) / (\\|u\|\\|v\|) | 🔴 Mesure pertinence |
| **FAISS IndexFlatIP** | Stockage et recherche rapide des vecteurs | 🔴 Infrastructure |
| **Inner Product** | = Cosine quand vecteurs normalisés | 🟡 Détail tech |
| **RRF** | Fusion BM25 + FAISS | 🟢 Optimisation |

---

## 📚 Ressources

- [Sentence Transformers docs](https://www.sbert.net/)
- [FAISS documentation](https://github.com/facebookresearch/faiss)
- [Cosine similarity on Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Vector databases guide](https://www.vector-database-guide.com/)

---

**Créé pour le pipeline RAG EvalLLM 2026** ✨
