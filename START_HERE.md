# 🚀 START HERE - Démarrage rapide (90 secondes)

## ⚡ TL;DR

**Trois commandes:**

```bash
# 1. Valider modèles HF online
python pipeline_hf.py --mode test

# 2. Indexer corpus (une fois)
python pipeline_hf.py --mode index \
    --corpus "/path/to/corpus" \
    --index ./index

# 3. Générer réponses
python pipeline_hf.py --mode run \
    --questions questions.json \
    --index ./index \
    --output soumission.json
```

---

## 📊 Workflow en 30 secondes

```
PDFs
  ↓ (extractor.py: page par page)
Chunks enrichis [DOC] [SECTION] [PAGE]
  ├─ Branche 1: BM25 (lexical) → bm25.pkl
  └─ Branche 2: Embeddings (semantic, 384D) → FAISS
       ↓ (indexer_hf_online.py)
Index complet: index/
  ├─ faiss.index (vecteurs)
  ├─ bm25.pkl
  ├─ chunks.json
  └─ mapping.json
       ↓
Requête → Encode (384D) 
       ├─ BM25 search
       ├─ FAISS search (cosine similarity)
       └─ RRF Fusion
            ↓
Top-K chunks
       ↓ (generator_hf.py)
Réponse JSON
```

**Temps:**
- Indexation: 100-500 ms
- Requête: 50-200 ms  
- Bout-en-bout: 1-5 s

---

## 🔍 Où sont les vecteurs?

**Stockage:**
```
index/faiss.index
├─ Type: IndexFlatIP (brute-force inner product = cosine similarity)
├─ Vecteurs: N chunks × 384 dimensions
├─ Format: Binary float32
├─ Taille: ~1.5 KB/chunk = 50-100 MB pour 5K pages
└─ Accès: O(n×d) ultra-optimisé C++
```

**Comment recherche:**
```python
# Pour chaque requête:
query_vec = model.encode(question)  # 384D normalized

# FAISS calcule:
scores = faiss_index.search(query_vec, k=10)
# similarity[i] = query · chunk[i]  (384 multiplications)
# Range: [0,1] pour vecteurs normalisés
```

**Interprétation de score:**
- 0.9-1.0 = très similaire (excellent match)
- 0.7-0.9 = similaire  (bon match)
- 0.5-0.7 = modérément similaire
- <0.5 = peu similaire

---

## 🔬 La science derrière

**Hybrid approach:**
- **BM25** (Robertson 1994): Lexical search, probabilistic
  - Excelle: termes rares, exactitude
  - Faible: paraphrase, synonyme
  
- **FAISS + Embeddings** (Karpukhin 2020): Semantic search
  - Excelle: paraphrase, contexte, sémantique
  - Faible: termes rares, stop words
  
- **RRF Fusion** (Cormack 2009): Combine sans hyperparamètres
  - Robustes même si l'une des sources est faible

**Cosine Similarity** (Salton 1983):
$$\cos(\theta) = u \cdot v = \sum_{i=1}^{384} u_i \times v_i \quad (|| u || = || v || = 1)$$

Géométrie: angle entre vecteurs dans espace 384D

---

## 🎯 Exemples rapides

### Test les démos

```bash
# Tests interactifs (math + vrai modèle)
python3 quick_test_vectors.py

# 6 démonstrations complètes
python3 demonstrate_vectors.py

# Inspecter l'index
python3 inspect_vectors.py --index ./index
```

### Voir structure du projet

```bash
ls -la *.py *.md *.sh
# 12 fichiers essentiels:
# - 4 CORE (extractor, indexer, generator, pipeline)
# - 4 TESTING (test, demo, inspect, quick_test)
# - 3 DOCS (README, START_HERE, AIDE_MEMOIRE)
# - 1 SCRIPT (quickstart.sh)
```

---

## 📖 Pour plus de détails

- **README.md** → Workflow complet + fondations scientifiques
- **AIDE_MEMOIRE_VECTEURS.md** → Formules mathématiques
- `demonstrate_vectors.py` → Voir code en action

---

## ✅ Checklist avant production

- [ ] `python pipeline_hf.py --mode test` → PASSE
- [ ] Corpus PDF préparé
- [ ] Questions JSON au format challenge
- [ ] Espace disque: > 100 MB
- [ ] Connexion Internet (télécharger modèles HF 1ère fois)

---

## 🚀 Démarrage recommandé

```bash
# Étape 1: Valider
python pipeline_hf.py --mode test

# Étape 2: Indexer (attendre)
python pipeline_hf.py --mode index \
    --corpus "/path/to/corpus" \
    --index ./index

# Étape 3: Tester requête
python pipeline_hf.py --mode run \
    --questions test_questions.json \
    --index ./index \
    --output test_output.json

# Étape 4: Production
python pipeline_hf.py --mode full \
    --corpus "/path/to/corpus" \
    --questions questions.json \
    --index ./index \
    --output soumission.json
```

---

**Prêt! Lancez:**
```bash
python pipeline_hf.py --mode test
```
