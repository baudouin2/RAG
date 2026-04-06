"""
indexer.py
==========
Construit et sauvegarde l'index hybride BM25 + FAISS à partir des chunks.

Architecture :
  - BM25  (rank_bm25)  : récupération lexicale, robuste aux termes rares
  - FAISS (faiss-cpu)  : récupération dense, robuste à la paraphrase sémantique
  - Fusion RRF         : Reciprocal Rank Fusion, sans hyperparamètre sensible
  - Reranker           : cross-encoder ms-marco-MiniLM (optionnel, ~22M params)

Invariant challenge : chaque résultat porte (doc_name, page) physique.
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
MODELE_EMBEDDING  = "all-MiniLM-L6-v2"         # 22M params, Apache 2.0, rapide CPU
MODELE_RERANKER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 22M params, Apache 2.0
DIM_EMBEDDING     = 384                          # dimension all-MiniLM-L6-v2
RRF_K             = 60                           # constante RRF (standard = 60)
TOP_K_INITIAL     = 20                           # nb candidats avant reranking
TOP_K_FINAL       = 10                           # nb résultats après reranking


# ──────────────────────────────────────────────
# Tokenisation légère pour BM25
# ──────────────────────────────────────────────
def tokeniser(texte: str) -> List[str]:
    """
    Tokenisation minimaliste : minuscules + split sur non-alphanumérique.
    Suffisant pour BM25Okapi. Pas besoin de NLTK ou spaCy.
    """
    import re
    tokens = re.split(r'[^a-zA-Z0-9àâäéèêëîïôùûüç]+', texte.lower())
    return [t for t in tokens if len(t) > 1]


# ──────────────────────────────────────────────
# Construction de l'index
# ──────────────────────────────────────────────
class IndexHybride:
    """
    Index hybride BM25 + FAISS avec Reciprocal Rank Fusion.

    Attributs publics :
        chunks       : liste de tous les chunks (dicts)
        chunks_valides : sous-liste des chunks effectivement indexés (est_vide=False)
        bm25         : index BM25Okapi
        faiss_index  : index FAISS (IndexFlatIP sur embeddings normalisés = cosine)
        embedder     : SentenceTransformer
        reranker     : CrossEncoder (None si désactivé)
    """

    def __init__(self, utiliser_reranker: bool = True):
        self.utiliser_reranker = utiliser_reranker
        self.chunks: List[Dict] = []
        self.chunks_valides: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        self._idx_valide_to_global: List[int] = []  # mapping idx_valide → idx_global

    def construire(self, chunks: List[Dict], batch_size: int = 64) -> None:
        """
        Construit BM25 + FAISS à partir de la liste de chunks.
        Les pages vides (est_vide=True) sont exclues de l'index mais conservées
        dans self.chunks pour la traçabilité.
        """
        self.chunks = chunks

        # Séparer chunks indexables des vides
        self._idx_valide_to_global = [
            i for i, c in enumerate(chunks) if not c["est_vide"]
        ]
        self.chunks_valides = [chunks[i] for i in self._idx_valide_to_global]

        print(f"\n[Indexation] {len(self.chunks_valides)} chunks valides "
              f"/ {len(self.chunks)} total")

        # ── BM25 ────────────────────────────────────────────────────────────
        print("[BM25] Tokenisation...")
        corpus_tokens = [
            tokeniser(c["texte_enrichi"]) for c in self.chunks_valides
        ]
        self.bm25 = BM25Okapi(corpus_tokens)
        print(f"[BM25] Index construit sur {len(corpus_tokens)} documents.")

        # ── FAISS ───────────────────────────────────────────────────────────
        print(f"[FAISS] Chargement modèle '{MODELE_EMBEDDING}'...")
        self.embedder = SentenceTransformer(MODELE_EMBEDDING)

        print(f"[FAISS] Encodage des chunks (batch_size={batch_size})...")
        textes = [c["texte_enrichi"] for c in self.chunks_valides]
        embeddings = self.embedder.encode(
            textes,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine via produit scalaire
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # IndexFlatIP : produit scalaire = cosine si vecteurs normalisés
        self.faiss_index = faiss.IndexFlatIP(DIM_EMBEDDING)
        self.faiss_index.add(embeddings)
        print(f"[FAISS] {self.faiss_index.ntotal} vecteurs indexés.")

        # ── Reranker ────────────────────────────────────────────────────────
        if self.utiliser_reranker:
            print(f"[Reranker] Chargement '{MODELE_RERANKER}'...")
            self.reranker = CrossEncoder(MODELE_RERANKER, max_length=512)
            print("[Reranker] Prêt.")

    # ──────────────────────────────────────────────
    # Sauvegarde / chargement
    # ──────────────────────────────────────────────
    def sauvegarder(self, dossier: Path) -> None:
        """Sauvegarde l'index complet sur disque."""
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)

        # FAISS
        faiss.write_index(self.faiss_index, str(dossier / "faiss.index"))

        # BM25 + métadonnées (pickle)
        with open(dossier / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        # Chunks (JSON)
        with open(dossier / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # Mapping idx_valide → global
        with open(dossier / "mapping.json", "w") as f:
            json.dump(self._idx_valide_to_global, f)

        print(f"[Save] Index sauvegardé dans {dossier}/")

    def charger(self, dossier: Path) -> None:
        """Recharge l'index depuis le disque."""
        dossier = Path(dossier)

        self.faiss_index = faiss.read_index(str(dossier / "faiss.index"))

        with open(dossier / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open(dossier / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        with open(dossier / "mapping.json", "r") as f:
            self._idx_valide_to_global = json.load(f)

        self.chunks_valides = [self.chunks[i] for i in self._idx_valide_to_global]

        print(f"[Load] Index chargé : {len(self.chunks_valides)} chunks valides.")

        print(f"[Load] Chargement modèle embedding '{MODELE_EMBEDDING}'...")
        self.embedder = SentenceTransformer(MODELE_EMBEDDING)

        if self.utiliser_reranker:
            print(f"[Load] Chargement reranker '{MODELE_RERANKER}'...")
            self.reranker = CrossEncoder(MODELE_RERANKER, max_length=512)


# ──────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────
def rrf_fusion(
    listes: List[List[int]],
    k: int = RRF_K,
) -> List[Tuple[int, float]]:
    """
    Fusionne plusieurs listes de rangs en un score RRF unique.

    RRF(d) = Σ 1 / (k + rang(d))
    Les indices sont des positions dans chunks_valides.

    Retourne : liste triée [(idx_valide, score_rrf), ...]
    """
    scores: Dict[int, float] = {}
    for liste in listes:
        for rang, idx in enumerate(liste):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rang + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────
# Retrieval complet (BM25 + FAISS + RRF + reranker)
# ──────────────────────────────────────────────
def rechercher(
    index: IndexHybride,
    requete: str,
    top_k: int = TOP_K_FINAL,
    top_k_initial: int = TOP_K_INITIAL,
) -> List[Dict]:
    """
    Pipeline de retrieval complet pour une requête.

    Étapes :
      1. BM25 → top_k_initial candidats
      2. Dense (FAISS) → top_k_initial candidats
      3. RRF fusion → liste unique
      4. Reranker cross-encoder (si disponible) → re-classement
      5. Retourne top_k résultats

    Chaque résultat est un dict avec :
      doc_name, page, section, score_rrf, score_reranker, rang
    """
    n_valides = len(index.chunks_valides)
    top_k_initial = min(top_k_initial, n_valides)

    # ── Étape 1 : BM25 ──────────────────────────────────────────────────────
    tokens_requete = tokeniser(requete)
    scores_bm25 = index.bm25.get_scores(tokens_requete)
    idx_bm25 = list(np.argsort(scores_bm25)[::-1][:top_k_initial])

    # ── Étape 2 : Dense FAISS ───────────────────────────────────────────────
    vecteur = index.embedder.encode(
        [requete],
        normalize_embeddings=True,
    ).astype(np.float32)
    _, idx_faiss_arr = index.faiss_index.search(vecteur, top_k_initial)
    idx_dense = [int(i) for i in idx_faiss_arr[0] if i >= 0]

    # ── Étape 3 : RRF ───────────────────────────────────────────────────────
    fusionnes = rrf_fusion([idx_bm25, idx_dense])

    # Prendre les top_k_initial candidats après fusion (avant reranking)
    candidats_idx = [idx for idx, _ in fusionnes[:top_k_initial]]
    candidats_rrf = {idx: score for idx, score in fusionnes[:top_k_initial]}

    # ── Étape 4 : Reranker ──────────────────────────────────────────────────
    if index.reranker is not None and candidats_idx:
        paires = [
            (requete, index.chunks_valides[idx]["texte_enrichi"])
            for idx in candidats_idx
        ]
        scores_reranker = index.reranker.predict(paires)
        candidats_avec_score = sorted(
            zip(candidats_idx, scores_reranker),
            key=lambda x: x[1],
            reverse=True,
        )
        candidats_finaux = candidats_avec_score[:top_k]
    else:
        candidats_finaux = [(idx, 0.0) for idx, _ in fusionnes[:top_k]]

    # ── Construction des résultats ──────────────────────────────────────────
    resultats = []
    for rang, (idx_valide, score_reranker) in enumerate(candidats_finaux):
        chunk = index.chunks_valides[idx_valide]
        resultats.append({
            "rank"           : rang + 1,
            "doc_name"       : chunk["doc_name"],
            "page"           : chunk["page"],
            "section"        : chunk["section"],
            "score_rrf"      : round(candidats_rrf.get(idx_valide, 0.0), 6),
            "score_reranker" : round(float(score_reranker), 4),
            "texte_brut"     : chunk["texte_brut"],   # pour le générateur
            "nb_pages_doc"   : chunk["nb_pages_doc"],
            "regime"         : chunk["regime"],
        })

    return resultats


if __name__ == "__main__":
    print("indexer.py chargé correctement.")
    print("Usage : from indexer import IndexHybride, rechercher")
