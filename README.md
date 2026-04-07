# 🔬 Pipeline RAG Hybride — Challenge EvalLLM 2026

## 📋 Vue générale du projet

**Objectif:** Système de Retrieval-Augmented Generation (RAG) hybride fusion lexical (BM25) et sémantique (FAISS) pour répondre à des questions basées sur corpus de documents PDF.

**Approche:** Architecture hybride sans modèle propriétaire, 100% Hugging Face open source, CPU-compatible.

---

## 🔄 Workflow complet avec I/O

### Phase 1️⃣: EXTRACTION (Pipeline d'entrée)

```
📄 CORPUS PDF
├─ fichier_1.pdf (42 pages)
├─ fichier_2.pdf (18 pages)
└─ fichier_N.pdf (varies)
     │
     └─→ [extractor.py]
        ├─ Entrée: PDFs bruts
        ├─ Opération: Extraction page-par-page
        │  • Parse text + métadonnées (auteur, structure)
        │  • Détecte sections par font-size heuristic
        │  • (voir: taille_corps_mediane baseline)
        │  • Enrichissement uniforme: [DOC] [SECTION] [PAGE]
        │  • Nettoyage artefacts PDF
        ├─ Sortie: Liste Python de 5,000+ chunks
        │  [
        │    {
        │      "doc_name": "article.pdf",
        │      "page": 5,
        │      "section": "Methodology",
        │      "texte_enrichi": "[DOC: article.pdf] [SECTION: Methodology] [PAGE: 5/42]\n...",
        │      "texte_brut": "...",
        │      "est_vide": false
        │    },
        │    ...
        │  ]
        └─ Stratégie: Page physique = chunk (granularité constante)
```

**Fondation scientifique:**
- Chunking uniforme évite biais par taille document [Robertson et al., 2009]
- Enrichissement contextuel améliore retrieval sémantique [Karpukhin et al., 2020]

---

### Phase 2️⃣: INDEXATION (Construction index)

```
📊 CHUNKS (5K+ entries)
     │
     └─→ [indexer_hf_online.py] - IndexHybride
        │
        ├─ BRANCHE 1: BM25 (Lexical Search)
        │  ├─ Entrée: Chunks bruts tokenisés
        │  ├─ Opération: Calcul IDF (Inverse Document Frequency)
        │  │  BM25(d,q) = Σ IDF(qi) × (f(qi,d) × (k1+1)) / (f(qi,d) + k1×(1-b+b×|d|/avgdl))
        │  │  [Robertson et al., 1994]
        │  ├─ Sortie: bm25.pkl (sérialisation rank_bm25)
        │  │  - Matrice creuse doc×term
        │  │  - ~1-5 MB pour 5K docs
        │  └─ Temps: ~100-500 ms
        │
        ├─ BRANCHE 2: FAISS + Embeddings (Semantic Search)
        │  ├─ Entrée: Chunks enrichis (texte + contexte)
        │  │
        │  ├─ ÉTAPE A: Encoding (SentenceTransformer)
        │  │  ├─ Modèle: all-MiniLM-L6-v2 (22M params)
        │  │  ├─ Architecture: Dual-encoder Transformer
        │  │  │  chunk → [tokenization] → [embedding layer] → [attention] 
        │  │  │       → [pooling] → [normalization] → vecteur 384D
        │  │  ├─ Sortie: matrice (N_chunks, 384) normalized float32
        │  │  │  - Norme L2 = 1.0 pour tout vecteur (important!)
        │  │  │  - ~1.5 KB par chunk
        │  │  ├─ Temps: ~0.5-1 ms/chunk CPU
        │  │  ├─ Citation: [Sentence-BERT, Reimers & Gurevych 2019]
        │  │  └─ Propriété: cos(u,v) = u·v quand ||u||=||v||=1
        │  │
        │  └─ ÉTAPE B: Stockage FAISS
        │     ├─ Index Type: IndexFlatIP
        │     │  (Flat = brute-force, IP = Inner Product = cosine similarity)
        │     ├─ Opération: Ajout vecteurs en matrice dense
        │     ├─ Sortie: faiss.index
        │     │  - Header: type, dim=384, ntotal=N_chunks, metric=IP
        │     │  - Data: flux binaire des vecteurs
        │     │  - ~1.5 KB/chunk, total ≈ 50-100 MB pour 5K
        │     ├─ Temps: ~10-50 ms
        │     └─ Citation: [FAISS, Johnson et al. 2019]
        │
        └─ SORTIE GLOBALE: Fichiers sauvegardés
           index/
           ├─ faiss.index          (vecteurs stockés)
           ├─ bm25.pkl             (modèle lexical)
           ├─ chunks.json          (métadonnées)
           └─ mapping.json         (chunk_id → faiss_index)
```

**Fondations scientifiques:**
- **BM25:** Robertson/Zaragoza (2009) - Modèle probabiliste lexical robuste
  - Capture termes rares bien, mais insensible à paraphrase
  
- **FAISS:** Johnson et al. (2019) - Indexation dense pour recherche vecteur
  - IndexFlatIP: comparaison par produit scalaire (= cosinus si normalisé)
  - Complexité: O(n×d) par requête mais ultra-optimisé en C++

- **Normalisation:** Critique pour cosinus similarity
  - $$||v|| = 1 \Rightarrow \cos(\theta) = u \cdot v$$

---

### Phase 3️⃣: RETRIEVAL (Recherche par requête)

```
❓ REQUÊTE UTILISATEUR
"Quels résultats montre cette étude?"
     │
     └─→ [indexer_hf_online.py] rechercher()
        │
        ├─ ÉTAPE 1: Encoding requête
        │  ├─ Input: Chaîne texte requête
        │  ├─ Process: SentenceTransformer.encode([query])
        │  ├─ Output: vecteur 384D normalisé
        │  └─ Temps: ~1-2 ms
        │
        ├─ ÉTAPE 2: Parallèle - BM25 Search
        │  ├─ Tokenization: "quels" "résultats" "montre" "étude"
        │  ├─ Opération: BM25 scoring sur tous les chunks
        │  │  → scores = bm25_index.get_scores(tokens_requete)
        │  ├─ Output: [chunk_id, score_bm25, rank_bm25] × top_k
        │  │  Ex: [(145, 8.73, 1), (89, 7.21, 2), (234, 6.45, 3), ...]
        │  └─ Temps: ~10-50 ms
        │
        ├─ ÉTAPE 3: Parallèle - FAISS Search  
        │  ├─ Opération: faiss_index.search(query_vec, k=top_k_initial)
        │  │  Algo: Brute-force inner product pour tous les N vecteurs
        │  │  similarity_score[i] = query_vec · chunk_vec[i]
        │  │  Range: [-1, 1] conventionnellement [0, 1] pour vecteurs+ normalisés
        │  ├─ Math: ∀i: sim[i] = Σ(q_d × c_d) pour d=1..384
        │  │  (384 multiplications + 383 additions par chunk)
        │  ├─ Output: [chunk_id, score_faiss, rank_faiss] × top_k
        │  │  Ex: [(234, 0.876, 1), (145, 0.834, 2), (567, 0.823, 3), ...]
        │  ├─ Temps: ~5-20 ms
        │  └─ Citation: [Cosinus similarity, Salton & McGill 1983]
        │
        ├─ ÉTAPE 4: Fusion RRF
        │  ├─ Entrée: 2 listes de rangs (BM25 et FAISS)
        │  ├─ Opération: Reciprocal Rank Fusion
        │  │  score_rrf[i] = Σ 1/(k + rank[i])  {k=60 standard}
        │  │  Combine rangs sans besoin d'hyperparamètres
        │  ├─ Output: Chunks fusion-triés, score décroissant
        │  │  Ex: [(234, 0.892), (145, 0.878), (89, 0.821), ...]
        │  └─ Citation: [Cormack et al., 2009]
        │
        └─ ÉTAPE 5: Optionnel - Reranking
           ├─ Modèle: cross-encoder ms-marco-MiniLM-L6-v2
           ├─ Entrée: Top-K chunks de RRF + requête
           ├─ Opération: Fine-tuning P(relevance|query, chunk)
           │  → Calcule score binaire/continu pour la paire (query, chunk)
           ├─ Output: Re-ranked chunks, ordre final confiance décroissante
           └─ Citation: [Dense Passage Retrieval, Karpukhin et al. 2020]

📤 SORTIE RETRIEVAL:
   [
     {
       "chunk_id": 234,
       "rank": 1,
       "score_rrf": 0.892,
       "doc_name": "study.pdf",
       "page": 12,
       "texte": "[DOC: study.pdf] [PAGE: 12/50]\n..."
     },
     ...
   ] × top_k (défaut: 10)
```

**Fondations scientifiques:**
- **BM25 + FAISS Hybrid:** Complémentarité
  - BM25 excelle: termes rares, stop words, exactitude
  - FAISS excelle: paraphrase, sémantique, contexte
  - Fusion = robustesse [Combining Approaches, Voorhees & Harman 2000]

- **RRF:** Sans hyperparamètres, fusion effective
  - Score = Σ 1/(k + rank) agnostique aux magnitudes de scores
  - Robuste même si l'une des sources est mauvaise

- **Cosinus Similarity:** Géométrie
  - Angle entre vecteurs dans espace sémantique 384D
  - 0.9+ = très similaire (même topic)
  - 0.5-0.8 = similaire (même domaine)
  - <0.5 = peu similaire

---

### Phase 4️⃣: GÉNÉRATION (Réponse finale)

```
📊 TOP-K CHUNKS (10 par défaut)
     │
     └─→ [generator_hf.py] generer_reponse()
        │
        ├─ MODE A: Extraction (par défaut, sans modèle)
        │  ├─ Entrée: Chunks + requête
        │  ├─ Opération:
        │  │  1. Tokenize chunks + requête
        │  │  2. Score chaque phrase par overlap mots-clés requête
        │  │  3. Sélectionner top-N phrases, ordre original
        │  │  4. Concaténer
        │  ├─ Sortie: Texte réponse < 500 mots
        │  ├─ Temps: ~50-200 ms
        │  └─ Avantage: Déterministe, sans hallucination
        │
        └─ MODE B: LLM (optionnel)
           ├─ Entrée: Chunks + requête
           ├─ Opération: Few-shot prompt → LLM inference
           │  (ex: via inférence.ai gratuit)
           ├─ Sortie: Réponse générée par LLM
           └─ Temps: ~2-10 s
```

**Fondation scientifique:**
- **Extraction vs. Génération:** Trade-off
  - Extraction: Déterministe, traçable, moins hallucination
    - Citation: [Extractive QA, SQuAD benchmark]
  - Génération: Créatif, cohérent, mais peut fabriquer
    - Citation: [Generative QA, T5/BART]

---

## 📊 Architecture scientifique complète

### Vue système

```
                   ┌──────────────────────────────────────┐
                   │    CORPUS PDF (N documents)          │
                   └──────────┬───────────────────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │   Extractor  │
                        │   (PyMuPDF)  │
                        └──────┬───────┘
                               │
                  ┌────────────┴────────────┐
                  │                        │
                  ▼                        ▼
        ┌─────────────────┐     ┌──────────────────────┐
        │  Tokenizer (BM25)│     │ SentenceTransformer  │
        │                 │     │  (all-MiniLM-L6-v2)  │
        └────────┬────────┘     └──────────┬───────────┘
                 │                        │
                 ▼                        ▼
        ┌──────────────┐        ┌──────────────────────┐
        │ BM25 Index   │        │   FAISS Index        │
        │ (bm25.pkl)   │        │  (faiss.index)       │
        │              │        │   384D vectors       │
        └──────┬───────┘        └──────────┬───────────┘
               │                          │
               │  ┌───────────────────────┘
               │  │
               │  ▼  [Requête encodée]
               │ ┌──────────────────┐
               │ │  Encode Query    │
               │ └────┬─────────────┘
               │      │
        ┌──────┴──────┴────────┐
        │                      │
        ▼                      ▼
    BM25 Scores         FAISS Scores
    (Lexical)           (Semantic)
        │                      │
        └──────────┬───────────┘
                   │
                   ▼
            ┌─────────────┐
            │ RRF Fusion  │
            │ (combine)   │
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────────────┐
            │ Optionnel: Reranker │
            │ (cross-encoder)     │
            └──────┬──────────────┘
                   │
                   ▼
            ┌──────────────────┐
            │  Top-K Chunks    │
            │ (merged & ranked) │
            └──────┬───────────┘
                   │
                   ▼
            ┌──────────────────┐
            │    Generator     │
            │  (Extraction or  │
            │      LLM)        │
            └──────┬───────────┘
                   │
                   ▼
            ┌──────────────────┐
            │  RÉPONSE JSON    │
            │  (EvalLLM 2026)  │
            └──────────────────┘
```

### Équations fondamentales

**1. BM25 (Probabilistic Ranking)**
$$\text{BM25}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i,D) \cdot (k_1 + 1)}{f(q_i,D) + k_1 \cdot (1-b+b \cdot \frac{|D|}{\text{avgdl}})}$$

Où:
- $\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}$
- $f(q_i,D)$ = fréquence terme dans doc
- $k_1, b$ = paramètres (généralement 2.0, 0.75)

**2. Dense Embeddings (Transformer)**
$$\text{embedding} = \text{normalize}(\text{PoolingLayer}(\text{TransformerEncoder}(\text{tokens})))$$

Sorties: vecteur 384D, $||v|| = 1$

**3. Cosinus Similarity (pour vecteurs normalisés)**
$$\text{similarity}(u,v) = u \cdot v = \sum_{i=1}^{384} u_i \times v_i \quad \in [-1, 1]$$

Interprétation:
- 0.9-1.0: Très similaire (même topic, paraphrase)
- 0.7-0.9: Similaire (related topics)
- 0.5-0.7: Modérément similaire
- <0.5: Peu similaire

**4. Reciprocal Rank Fusion**
$$\text{score}_\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

Où $R$ = rangements (BM25, FAISS), $k = 60$ (standard), $r(d)$ = rang du doc $d$

---

## 🎯 Propriétés du système

| Propriété | Valeur | Justification |
|-----------|--------|---------------|
| **Recall** | Haute (FAISS sémantique) | Capture paraphrase, synonyme |
| **Precision** | Moyenne-Haute (fusion RRF) | BM25 filtre bruit sémantique |
| **Latence requête** | 50-200 ms | FAISS brute-force très rapide CPU |
| **Taille index** | ~1.5 KB/chunk | Vecteurs 384D float32 compacts |
| **Pas d'hallucination** | ✓ (mode extraction) | Extraction de chunks seulement |
| **Pas de dépendance** | ✓ (HF online) | Apache 2.0 models, auto-DL |

---

## 📌 Configuration et usage

### Installation

```bash
pip install pymupdf rank_bm25 faiss-cpu sentence-transformers numpy
```

### Workflow complet

```bash
# 1. Test modèles HF
python pipeline_hf.py --mode test

# 2. Indexation (une fois)
python pipeline_hf.py --mode index \
  --corpus /path/to/corpus \
  --index ./index

# 3. Réponses aux questions
python pipeline_hf.py --mode run \
  --questions questions.json \
  --index ./index \
  --output responses.json

# Ou pipeline complet
python pipeline_hf.py --mode full \
  --corpus /path/to/corpus \
  --questions questions.json \
  --index ./index \
  --output responses.json
```

---

## 🧪 Tests et validation

```bash
# Tests interactifs (maths + FAISS + vrai modèle)
python3 quick_test_vectors.py

# 6 démonstrations complètes
python3 demonstrate_vectors.py

# Inspecter index
python3 inspect_vectors.py --index ./index

# Tests unitaires
python3 test_pipeline.py
```

---

## 📖 Documentation complémentaire

- **START_HERE.md** — Tutorial rapide (30s)
- **AIDE_MEMOIRE_VECTEURS.md** — Formules & quick reference

---

## 📚 Références académiques

- Salton & McGill (1983): "Introduction to Modern Information Retrieval"
- Robertson & Zaragoza (2009): "The Probabilistic Relevance Framework (BM25)" 
- Johnson et al. (2019): "FAISS: A Library for Efficient Similarity Search"
- Reimers & Gurevych (2019): "Sentence-BERT (SBERT)"
- Karpukhin et al. (2020): "Dense Passage Retrieval for Open-Domain QA"
- Cormack et al. (2009): "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"

---

**Status:** ✅ Production-ready (EvalLLM 2026)
