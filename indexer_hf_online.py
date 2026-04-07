"""
indexer_hf_online.py
====================
Index hybride BM25 + FAISS — 100% Hugging Face ONLINE & GRATUIT

✓ Auto-télécharge les modèles au 1er usage (~1-2 min)
✓ Zéro API key, zéro dépendance externe requise
✓ Apache 2.0, open source complet
✓ Validation des modèles pour garantir fonctionnalité
"""

import json
import pickle
import numpy as np
import faiss
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ══════════════════════════════════════════════════════════════════
# CONSTANTES — Modèles Hugging Face 100% ONLINE & GRATUITS
# ══════════════════════════════════════════════════════════════════

# Embeddings Apache 2.0 (auto-téléchargement au 1er usage)
MODELE_EMBEDDING_DEFAUT  = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, ⭐ rapidité+précision
DIM_EMBEDDING_DEFAUT     = 384

# Reranker Apache 2.0 (ultra-frugal, 22M)
MODELE_RERANKER_DEFAUT   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Fusion et retrieval
RRF_K             = 60                           # standard RRF
TOP_K_INITIAL     = 20                           # nb candidats avant reranking
TOP_K_FINAL       = 10                           # nb résultats après reranking  
MAX_CHUNKS_PER_DOC = 2                           # équité entre documents


# ══════════════════════════════════════════════════════════════════
# TOKENISATION BM25
# ══════════════════════════════════════════════════════════════════

def tokeniser(texte: str) -> List[str]:
    """Tokenisation légère pour BM25."""
    tokens = re.split(r'[^a-zA-Z0-9àâäéèêëîïôùûüç]+', texte.lower())
    return [t for t in tokens if len(t) > 1]


# ══════════════════════════════════════════════════════════════════
# VALIDATION DES MODÈLES HUGGING FACE
# ══════════════════════════════════════════════════════════════════

def valider_embedding(modele: str = MODELE_EMBEDDING_DEFAUT) -> bool:
    """
    Teste si le modèle embedding est accessible et fonctionnel.
    ✓ Télécharge au 1er usage (~30-60s)
    ✓ Utilise le cache après
    """
    try:
        print(f"  ▶ Embedding '{modele}'...", end=" ", flush=True)
        st = SentenceTransformer(modele)
        
        # Test d'encoding
        test_emb = st.encode(["test de validation"], normalize_embeddings=True)
        dims = st.get_sentence_embedding_dimension()
        
        assert test_emb.shape == (1, dims), f"Shape error: {test_emb.shape}"
        print(f"✓ ({dims} dims)")
        return True
    except Exception as e:
        print(f"✗ ERREUR: {str(e)[:80]}")
        return False


def valider_reranker(modele: str = MODELE_RERANKER_DEFAUT) -> bool:
    """
    Teste si le modèle reranker est accessible et fonctionnel.
    ✓ Télécharge au 1er usage (~20-40s)
    ✓ Utilise le cache après
    """
    try:
        print(f"  ▶ Reranker '{modele}'...", end=" ", flush=True)
        ce = CrossEncoder(modele, max_length=512)
        
        # Test de scoring
        scores = ce.predict([["query test", "document test"]])
        
        assert len(scores) == 1, "Dim error"
        print(f"✓ (score: {scores[0]:.2f})")
        return True
    except Exception as e:
        print(f"✗ ERREUR: {str(e)[:80]}")
        return False


def test_modeles_startup(embedding: str = MODELE_EMBEDDING_DEFAUT,
                         reranker: str = MODELE_RERANKER_DEFAUT) -> bool:
    """
    Test de démarrage pour valider que tous les modèles sont accessibles.
    À appeler UNE FOIS au startup du pipeline.
    
    Retour:
      True si OK, False si erreur
    """
    print("\n" + "="*70)
    print("🔌 VALIDATION DES MODÈLES HUGGING FACE — MODE ONLINE")
    print("="*70)
    print("\n[1/2] Test de téléchargement - première utilisation (~2min)...")
    
    ok_emb = valider_embedding(embedding)
    ok_re = valider_reranker(reranker)
    
    print()
    if ok_emb and ok_re:
        print("✓ SUCCÈS — Tous les modèles sont fonctionnels!")
        print("  └─ Prochaines exécutions utiliseront le cache local")
        print("="*70 + "\n")
        return True
    else:
        print("✗ ERREUR — Un ou plusieurs modèles ne sont pas accessibles")
        print("  └─ Vérifiez votre connexion internet")
        print("  └─ Vérifiez que Hugging Face Hub est accessible")
        print("="*70 + "\n")
        return False


# ══════════════════════════════════════════════════════════════════
# INDEX HYBRIDE BM25 + FAISS + RRF + RERANKER
# ══════════════════════════════════════════════════════════════════

class IndexHybride:
    """
    Index hybride BM25 + FAISS avec Reciprocal Rank Fusion.
    100% Hugging Face, mode online, zéro API key.
    """

    def __init__(self,
                 utiliser_reranker: bool = True,
                 modele_embedding: str = MODELE_EMBEDDING_DEFAUT,
                 modele_reranker: str = MODELE_RERANKER_DEFAUT):
        
        self.utiliser_reranker = utiliser_reranker
        self.modele_embedding = modele_embedding
        self.modele_reranker = modele_reranker
        
        self.chunks: List[Dict] = []
        self.chunks_valides: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        self._idx_valide_to_global: List[int] = []

    def construire(self, chunks: List[Dict], batch_size: int = 64) -> None:
        """Construit l'index BM25 + FAISS."""
        self.chunks = chunks
        
        # Séparer chunks indexables des vides
        self._idx_valide_to_global = [
            i for i, c in enumerate(chunks) if not c["est_vide"]
        ]
        self.chunks_valides = [chunks[i] for i in self._idx_valide_to_global]

        print(f"\n[Indexation] {len(self.chunks_valides)} chunks valides / {len(self.chunks)} total")

        # ── BM25 ────────────────────────────────────────────────────────────
        print("[BM25] Construction...", end=" ", flush=True)
        corpus_tokens = [tokeniser(c["texte_enrichi"]) for c in self.chunks_valides]
        self.bm25 = BM25Okapi(corpus_tokens)
        print(f"✓ ({len(corpus_tokens)} docs)")

        # ── FAISS embedding ──────────────────────────────────────────────────
        print(f"[Embedding] Chargement HF '{self.modele_embedding}'...", end=" ", flush=True)
        self.embedder = SentenceTransformer(self.modele_embedding)
        print("✓")

        print(f"[Embedding] Encodage {len(self.chunks_valides)} chunks...", end=" ", flush=True)
        textes = [c["texte_enrichi"] for c in self.chunks_valides]
        embeddings = self.embedder.encode(
            textes,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        print("✓")

        # ── FAISS index ──────────────────────────────────────────────────────
        dim = self.embedder.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        print(f"[FAISS] Index construit ✓ ({dim} dims, {self.faiss_index.ntotal} vecteurs)")

        # ── Reranker ─────────────────────────────────────────────────────────
        if self.utiliser_reranker:
            print(f"[Reranker] Chargement HF '{self.modele_reranker}'...", end=" ", flush=True)
            self.reranker = CrossEncoder(self.modele_reranker, max_length=512)
            print("✓")
        else:
            print("[Reranker] Désactivé (mode rapide)")

    def sauvegarder(self, dossier: Path) -> None:
        """Sauvegarde l'index sur disque."""
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

        # Mapping
        with open(dossier / "mapping.json", "w") as f:
            json.dump(self._idx_valide_to_global, f)

        print(f"[Save] Index sauvegardé → {dossier}/")

    def charger(self, dossier: Path) -> None:
        """Charge l'index depuis le disque."""
        dossier = Path(dossier)
        
        if not dossier.exists():
            raise FileNotFoundError(f"Dossier d'index introuvable: {dossier}")
        
        fichiers_requis = ["faiss.index", "bm25.pkl", "chunks.json", "mapping.json"]
        fichiers_manquants = [f for f in fichiers_requis if not (dossier / f).exists()]
        
        if fichiers_manquants:
            raise FileNotFoundError(f"Fichiers d'index manquants: {fichiers_manquants}")
        
        print(f"[Load] Chargement de l'index depuis {dossier}/...")
        
        # Charger FAISS
        print("  [FAISS]", end=" ", flush=True)
        self.faiss_index = faiss.read_index(str(dossier / "faiss.index"))
        print(f"✓ ({self.faiss_index.ntotal} vecteurs)")
        
        # Charger BM25
        print("  [BM25]", end=" ", flush=True)
        with open(dossier / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        print("✓")
        
        # Charger chunks
        print("  [Chunks]", end=" ", flush=True)
        with open(dossier / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"✓ ({len(self.chunks)} total)")
        
        # Charger mapping
        print("  [Mapping]", end=" ", flush=True)
        with open(dossier / "mapping.json", "r") as f:
            self._idx_valide_to_global = json.load(f)
        self.chunks_valides = [self.chunks[i] for i in self._idx_valide_to_global]
        print(f"✓ ({len(self.chunks_valides)} valides)")
        
        # Charger le modèle embedding
        print(f"  [Embedding] Chargement HF '{self.modele_embedding}'...", end=" ", flush=True)
        self.embedder = SentenceTransformer(self.modele_embedding)
        print("✓")
        
        # Charger le reranker si activé
        if self.utiliser_reranker:
            print(f"  [Reranker] Chargement HF '{self.modele_reranker}'...", end=" ", flush=True)
            self.reranker = CrossEncoder(self.modele_reranker, max_length=512)
            print("✓")
        
        print("[Load] Index chargé ✓\n")


# ══════════════════════════════════════════════════════════════════
# RECIPROCAL RANK FUSION (RRF)
# ══════════════════════════════════════════════════════════════════

def rrf_fusion(listes: List[List[int]], k: int = RRF_K) -> List[Tuple[int, float]]:
    """Fusionne plusieurs rangs via RRF sans hyperparamètre sensible."""
    scores: Dict[int, float] = {}
    for liste in listes:
        for rang, idx in enumerate(liste):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rang)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ══════════════════════════════════════════════════════════════════
# RETRIEVAL COMPLET
# ══════════════════════════════════════════════════════════════════

def rechercher(index: IndexHybride,
               requete: str,
               top_k: int = TOP_K_FINAL,
               top_k_initial: int = TOP_K_INITIAL) -> List[Dict]:
    """
    Pipeline complet de retrieval:
      1. BM25 lexical
      2. FAISS dense (embedding HF)
      3. RRF fusion
      4. Reranker cross-encoder (optionnel)
    """
    
    n_valides = len(index.chunks_valides)
    top_k_initial = min(top_k_initial, n_valides)

    # ─ BM25  ────────────────────────────────────────────────────────
    tokens_requete = tokeniser(requete)
    scores_bm25 = index.bm25.get_scores(tokens_requete)
    idx_bm25 = list(np.argsort(scores_bm25)[::-1][:top_k_initial])

    # ─ FAISS ────────────────────────────────────────────────────────
    vecteur = index.embedder.encode(
        [requete],
        normalize_embeddings=True,
    ).astype(np.float32)
    _, idx_faiss_arr = index.faiss_index.search(vecteur, top_k_initial)
    idx_dense = [int(i) for i in idx_faiss_arr[0] if i >= 0]

    # ─ RRF fusion ───────────────────────────────────────────────────
    fusionnes = rrf_fusion([idx_bm25, idx_dense])
    candidats_idx = [idx for idx, _ in fusionnes[:top_k_initial]]
    candidats_rrf = {idx: score for idx, score in fusionnes[:top_k_initial]}

    # ─ Reranker ─────────────────────────────────────────────────────
    if index.reranker is not None and candidats_idx:
        pairs = [[requete, index.chunks_valides[idx]["texte_brut"][:512]]
                 for idx in candidats_idx]
        scores_ce = index.reranker.predict(pairs)
        candidats_finaux = sorted(
            enumerate(scores_ce),
            key=lambda x: x[1],
            reverse=True
        )
        candidats_finaux = [(candidats_idx[i], s) for i, s in candidats_finaux[:top_k]]
    else:
        candidats_finaux = [(idx, candidats_rrf[idx]) for idx, _ in fusionnes[:top_k]]

    # ─ Construction résultats ───────────────────────────────────────
    resultats = []
    for rang, (idx_valide, score_reranker) in enumerate(candidats_finaux):
        idx_global = index._idx_valide_to_global[idx_valide]
        chunk = index.chunks_valides[idx_valide]
        
        resultats.append({
            "rank": rang + 1,
            "doc_name": chunk["doc_name"],
            "page": chunk["page"],
            "section": chunk.get("section", ""),
            "score_rrf": candidats_rrf.get(idx_valide, 0.0),
            "score_reranker": float(score_reranker) if index.reranker else 0.0,
            "nb_pages_doc": chunk["nb_pages_doc"],
            "regime": chunk.get("regime", ""),
            "texte_brut": chunk["texte_brut"],
        })

    return resultats


if __name__ == "__main__":
    print("indexer_hf_online.py: 100% Hugging Face ONLINE")
    print("Usage: from indexer_hf_online import IndexHybride, rechercher, test_modeles_startup")
