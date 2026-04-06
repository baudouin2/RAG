# Pipeline RAG — Challenge EvalLLM 2026
## Architecture hybride frugale et open source

Stack : PyMuPDF · BM25 · FAISS · sentence-transformers · cross-encoder

---

## Structure du projet

```
rag_challenge/
├── extractor.py     # Extraction PDF + enrichissement contextuel adaptatif
├── indexer.py       # Index hybride BM25 + FAISS + RRF + reranker
├── generator.py     # Génération de réponses + formatage JSON challenge
├── pipeline.py      # Point d'entrée CLI (index / run / full)
├── test_pipeline.py # Tests de validation avec PDFs synthétiques
├── corpus/          # → Placer vos PDFs ici
├── index/           # → Index généré automatiquement
└── output/          # → Soumissions JSON générées
```

---

## Installation

```bash
pip install pymupdf rank_bm25 faiss-cpu sentence-transformers numpy
# Optionnel pour le mode LLM :
# ollama pull qwen2.5:7b
```

---

## Usage rapide

### 1. Indexer le corpus (une seule fois)
```bash
python pipeline.py --mode index --corpus ./corpus --index ./index
```

### 2. Répondre aux questions du challenge
```bash
python pipeline.py --mode run \
    --questions questions_challenge.json \
    --index ./index \
    --output soumission.json
```

### 3. Pipeline complet (index + run)
```bash
python pipeline.py --mode full \
    --corpus ./corpus \
    --questions questions_challenge.json \
    --output soumission.json
```

### Options
| Option | Défaut | Description |
|---|---|---|
| `--top_k` | 10 | Pages retournées par requête |
| `--no-rerank` | False | Désactive le reranker (plus rapide) |
| `--generation` | extraction | `extraction` (sans GPU) ou `llm` (Ollama) |

---

## Stratégie de chunking adaptatif

| Régime | Pages | Stratégie | Justification |
|---|---|---|---|
| Court | 1–3 | Document entier | Sur-découpage nuit à l'embedding |
| Moyen | 4–15 | Page physique | Granularité optimale (NVIDIA 2024) |
| Long | >15 | Page + overlap 200 tokens | Continuité inter-pages préservée |

**Enrichissement uniforme** (tous régimes) :
```
[DOC: fichier.pdf] [SECTION: titre détecté] [PAGE: n/total]
texte de la page...
```

---

## Architecture du retrieval

```
Requête
  │
  ├─► BM25 (lexical)  ──┐
  │                     ├─► RRF Fusion ──► Reranker ──► Top-K
  └─► FAISS (dense) ───┘                 cross-encoder    résultats
```

- **BM25** : `rank_bm25`, tokenisation légère, robuste aux termes rares
- **FAISS** : `IndexFlatIP`, embeddings `all-MiniLM-L6-v2` normalisés (cosine)
- **RRF** : `score = Σ 1/(60 + rang)`, sans hyperparamètre sensible
- **Reranker** : `cross-encoder/ms-marco-MiniLM-L-6-v2`, 22M params, CPU-friendly

---

## Format de sortie JSON (challenge EvalLLM 2026)

```json
{
  "run_id": "RAG-Challenge-EvalLLM2026",
  "parameters": {
    "modele_embedding": "all-MiniLM-L6-v2",
    "top_k_final": 10,
    "rrf_k": 60,
    "overlap_tokens": 200
  },
  "results": [
    {
      "qid": "Q1",
      "question": "...",
      "retrieved": [
        {
          "rank": 1,
          "doc_name": "document.pdf",
          "page": 7,
          "metadata": {
            "section": "3.2 Installation",
            "score_rrf": 0.0312,
            "score_reranker": 4.231,
            "nb_pages_doc": 42,
            "regime": "long"
          }
        }
      ],
      "answer": "Réponse synthétisée...",
      "metadata": {}
    }
  ]
}
```

---

## Mode génération LLM (optionnel)

Nécessite [Ollama](https://ollama.ai) installé localement :

```bash
# Installer Ollama puis :
ollama pull qwen2.5:7b          # 4.7 GB, Apache 2.0
ollama serve                     # démarre le serveur local

# Lancer le pipeline en mode LLM
python pipeline.py --mode run \
    --questions questions.json \
    --output soumission.json \
    --generation llm
```

Modèles alternatifs (Apache 2.0) :
- `mistral:7b` (~4 GB)
- `qwen2.5:3b` (~2 GB, plus frugal)
- `phi3:mini` (~2.3 GB, très frugal)

---

## Métriques cibles (challenge)

| Métrique | Composant | Stratégie d'optimisation |
|---|---|---|
| Précision | Retrieval | Reranker cross-encoder |
| Rappel | Retrieval | RRF (BM25 + dense) + top_k ≥ 10 |
| NDCG | Retrieval | Fusion + reranking par pertinence |
| BERTScore | Génération | Contexte enrichi + réponse ancrée |
| LLM-Judge | Génération | Mode LLM Qwen2.5-7B recommandé |
| Bonus OS | Global | 100% open source (Apache 2.0) ✓ |
| Bonus frugal | Global | SLM ≤ 7B, CPU-friendly ✓ |

---

## Licences

| Composant | Licence |
|---|---|
| PyMuPDF | AGPL-3.0 (ou commercial) |
| rank_bm25 | Apache 2.0 |
| FAISS | MIT |
| all-MiniLM-L6-v2 | Apache 2.0 |
| ms-marco-MiniLM-L-6-v2 | Apache 2.0 |
| Qwen2.5-7B | Apache 2.0 |
| Mistral-7B | Apache 2.0 |
# RAG
# RAG
