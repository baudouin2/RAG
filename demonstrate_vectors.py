#!/usr/bin/env python3
"""
demonstrate_vectors.py
======================
Démonstration complète du stockage des vecteurs et du calcul de similarité cosinus.

Montre:
1. Comment les vecteurs sont créés (embeddings)
2. Comment ils sont stockés (FAISS)
3. Comment la similarité cosinus est calculée
4. Visualisation pratique
"""

import numpy as np
import json
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 1 : CRÉATION DES EMBEDDINGS
# ──────────────────────────────────────────────────────────────────────────────

def demo_embedding_creation():
    """
    Montre comment un texte devient un vecteur (embedding).
    """
    print("\n" + "="*80)
    print("PARTIE 1 : CRÉATION DES EMBEDDINGS (Texte → Vecteurs)")
    print("="*80)
    
    # Simpler le modèle all-MiniLM-L6-v2 avec 384 dimensions
    print("\n[Modèle] all-MiniLM-L6-v2")
    print("  - Paramètres : 22 millions")
    print("  - Dimensions : 384 (chaque embedding = 384 nombres)")
    print("  - Poids : réseau de neurones profond (Transformer)")
    
    # Exemple 1
    print("\n[Exemple 1] Premier chunk:")
    texte1 = "[DOC: article.pdf] [SECTION: Introduction] [PAGE: 1/10]\nCette étude examine les résultats de l'intelligence artificielle."
    print(f"  Texte: \"{texte1[:80]}...\"")
    print(f"  Longueur texte: {len(texte1)} caractères")
    
    # Simulation d'embedding (en vrai, c'est fait par SentenceTransformer)
    np.random.seed(42)  # Déterministe pour la démo
    embedding1 = np.random.randn(384).astype(np.float32)
    embedding1 = embedding1 / np.linalg.norm(embedding1)  # Normalisation (important!)
    
    print(f"\n  Embedding généré:")
    print(f"    [{embedding1[0]:8.5f}, {embedding1[1]:8.5f}, {embedding1[2]:8.5f}, ... ({384} dimensions total)]")
    print(f"    Type: float32")
    print(f"    Taille mémoire: {embedding1.nbytes} bytes = {embedding1.nbytes/1024:.2f} KB")
    print(f"    Nombre de valeurs: {len(embedding1)}")
    print(f"    Plage de valeurs: [{embedding1.min():.4f}, {embedding1.max():.4f}]")
    print(f"    Norme (||v||): {np.linalg.norm(embedding1):.6f} ← NORMALISÉ (important pour cosine)")
    
    # Exemple 2
    print("\n[Exemple 2] Deuxième chunk:")
    texte2 = "[DOC: article.pdf] [SECTION: Résultats] [PAGE: 5/10]\nLes résultats de notre étude montrent une amélioration significative."
    print(f"  Texte: \"{texte2[:80]}...\"")
    
    np.random.seed(43)
    embedding2 = np.random.randn(384).astype(np.float32)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    print(f"  Embedding généré: {embedding2.nbytes} bytes")
    print(f"    [{embedding2[0]:8.5f}, {embedding2[1]:8.5f}, {embedding2[2]:8.5f}, ...]")
    
    return embedding1, embedding2, texte1, texte2


# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 2 : STOCKAGE DANS FAISS
# ──────────────────────────────────────────────────────────────────────────────

def demo_faiss_storage(embeddings_list):
    """
    Montre comment les vecteurs sont stockés dans FAISS.
    """
    print("\n" + "="*80)
    print("PARTIE 2 : STOCKAGE DANS FAISS (faiss.index)")
    print("="*80)
    
    try:
        import faiss
    except ImportError:
        print("❌ FAISS non installé. Skipping FAISS storage demo.")
        return None
    
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    print("\n[Structure en mémoire de faiss.index]")
    print(f"""
┌─────────────────────────────────────────────────┐
│          FAISS Index (IndexFlatIP)              │
├─────────────────────────────────────────────────┤
│ Type: IndexFlatIP (Inner Product = cosine)     │
│ Dimension (d): {embeddings_array.shape[1]}                        │
│ Nombre de vecteurs (ntotal): {embeddings_array.shape[0]}              │
│                                                 │
│ Stockage:                                       │
│ ├─ Vecteur 0 : [{embeddings_array[0,0]:8.5f}, {embeddings_array[0,1]:8.5f}, ...]      │
│ ├─ Vecteur 1 : [{embeddings_array[1,0]:8.5f}, {embeddings_array[1,1]:8.5f}, ...]      │
│ ├─ Vecteur 2 : [{embeddings_array[2,0]:8.5f}, {embeddings_array[2,1]:8.5f}, ...]      │
│ └─ ... (total: {embeddings_array.shape[0]} vecteurs)         │
│                                                 │
│ Taille mémoire:                                 │
│   {embeddings_array.nbytes} bytes = {embeddings_array.nbytes / (1024*1024):.2f} MB           │
│   Par vecteur:                                  │
│     {embeddings_array.shape[1]} floats × 4 bytes = {embeddings_array.shape[1] * 4} bytes   │
└─────────────────────────────────────────────────┘
""")
    
    # Créer l'index FAISS
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    print(f"[FAISS Index créé]")
    print(f"  ✓ Dimension acceptée: {dimension}")
    print(f"  ✓ Vecteurs ajoutés: {index.ntotal}")
    print(f"  ✓ Index opérationnel et prêt pour les recherches")
    
    return index, embeddings_array


# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 3 : CALCUL DE SIMILARITÉ COSINUS
# ──────────────────────────────────────────────────────────────────────────────

def demo_cosine_similarity(embedding1, embedding2):
    """
    Montre le calcul détaillé de la similarité cosinus.
    """
    print("\n" + "="*80)
    print("PARTIE 3 : CALCUL DE SIMILARITÉ COSINUS")
    print("="*80)
    
    print("\n[Formule mathématique]")
    print("""
    Similarité cosinus = (u · v) / (||u|| × ||v||)
    
    Avec:
      u · v     = produit scalaire = Σ u[i] × v[i]
      ||u||     = norme de u = √(Σ u[i]²)
      ||v||     = norme de v = √(Σ v[i]²)
      
    Si vecteurs NORMALISÉS:
      ||u|| = 1 et ||v|| = 1
      → Similarité = u · v directement !
""")
    
    # Vérifier que les vecteurs sont normalisés
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    print(f"\n[Vérification normalisation]")
    print(f"  Norme embedding1: {norm1:.8f} (≈ 1.0 ? {abs(norm1 - 1.0) < 0.001})")
    print(f"  Norme embedding2: {norm2:.8f} (≈ 1.0 ? {abs(norm2 - 1.0) < 0.001})")
    
    # Calcul produit scalaire
    dot_product = np.dot(embedding1, embedding2)
    print(f"\n[Calcul étape par étape]")
    print(f"  Produit scalaire (Σ u[i] × v[i]):")
    print(f"    = {embedding1[0]:.5f} × {embedding2[0]:.5f} +")
    print(f"      {embedding1[1]:.5f} × {embedding2[1]:.5f} +")
    print(f"      {embedding1[2]:.5f} × {embedding2[2]:.5f} +")
    print(f"      ... (somme de 384 produits)")
    print(f"    = {dot_product:.8f}")
    
    # Finalisation
    print(f"\n  Comme les vecteurs sont NORMALISÉS:")
    print(f"    Similarité = {dot_product:.8f} / ({norm1:.6f} × {norm2:.6f})")
    print(f"              = {dot_product:.8f} / 1.0")
    print(f"              = {dot_product:.8f}")
    
    similarity_cosine = dot_product / (norm1 * norm2)
    
    print(f"\n[Résultat final]")
    print(f"  ✓ Similarité cosinus = {similarity_cosine:.8f}")
    print(f"  ✓ Plage: [-1, 1]")
    print(f"     -1.0 = complètement opposé")
    print(f"      0.0 = orthogonal (non lié)")
    print(f"      1.0 = identique")
    
    interpretation = "TRÈS BON MATCH 🎯" if similarity_cosine > 0.7 else \
                     "BON MATCH 👍" if similarity_cosine > 0.5 else \
                     "MATCH MOYEN" if similarity_cosine > 0.3 else \
                     "FAIBLE MATCH"
    print(f"  ✓ Interprétation: {interpretation}")
    
    return similarity_cosine


# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 4 : RECHERCHE AVEC REQUÊTE
# ──────────────────────────────────────────────────────────────────────────────

def demo_query_search(index, embeddings_array):
    """
    Montre comment une requête est transformée en vecteur et comparée.
    """
    print("\n" + "="*80)
    print("PARTIE 4 : RECHERCHE AVEC UNE REQUÊTE")
    print("="*80)
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers non installé. Skipping query demo.")
        return
    
    # Charger le modèle réel (une seule fois)
    print("\n[Chargement du modèle all-MiniLM-L6-v2]")
    print("  (première fois: télécharge ~90 MB de HuggingFace)")
    print("  (fois suivantes: charge du cache local ~/.cache/huggingface/)")
    
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"  ⚠️ Impossible de charger le modèle: {e}")
        print("  Utilisation d'embeddings simulés pour la démo")
        embedder = None
    
    # Requête
    query = "Quels sont les résultats de l'étude?"
    print(f"\n[Requête utilisateur]")
    print(f"  \"{query}\"")
    
    if embedder:
        # Encoder la requête avec le même modèle
        print(f"\n[Encodage de la requête]")
        query_embedding = embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        print(f"  Vecteur requête obtenu: {query_embedding.shape[0]} dimensions")
        print(f"  Norme: {np.linalg.norm(query_embedding):.8f}")
    else:
        # Simulation
        np.random.seed(44)
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Recherche avec FAISS
    print(f"\n[Recherche dans FAISS]")
    print(f"  Comparaison du vecteur requête avec ALL {embeddings_array.shape[0]} vecteurs")
    print(f"  Calcul: for each chunk: score = vecteur_requête · vecteur_chunk")
    
    if index is not None:
        distances, indices = index.search(query_embedding.reshape(1, -1), k=min(5, embeddings_array.shape[0]))
        distances = distances[0]
        indices = indices[0]
        
        print(f"\n[Résultats (triés par similarité décroissante)]")
        for rank, (idx, dist) in enumerate(zip(indices, distances)):
            bar_length = int(dist * 50)  # Pour la visualisation
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  Rang {rank+1}: Chunk {idx} | Score: {dist:.6f} | {bar}")
    
    print(f"\n[Comment FAISS fonctionne]")
    print(f"""
  1. Stockage (une fois):
     - Tous les vecteurs de chunks → mémoire RAM
     - Index FAISS optimise la structure pour recherche rapide
  
  2. À chaque requête:
     - Requête → transformée en vecteur (384 dimensions)
     - FAISS calcule: score = requête · chunk pour chaque chunk
     - Retour: top-K chunks avec meilleurs scores
  
  3. Temps:
     - Encodage requête: ~50 ms
     - Recherche FAISS: ~1-5 ms (pour 1000-10000 chunks)
     - Total: 50-55 ms (une fois les chunks indexés)
""")


# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 5 : VISUALISATION INTERNE
# ──────────────────────────────────────────────────────────────────────────────

def demo_vector_visualization():
    """
    Montre comment visualiser les vecteurs et comprendre leur structure.
    """
    print("\n" + "="*80)
    print("PARTIE 5 : VISUALISATION INTERNE DES VECTEURS")
    print("="*80)
    
    # Créer des vecteurs de démo
    np.random.seed(100)
    vectors = np.random.randn(5, 384).astype(np.float32)
    # Normaliser
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    print("\n[Matrice des vecteurs (première colonne = dimensions 0-10)]")
    print(f"""
    Chunk\Dim   0      1      2      3      4      5      6      7      8      9     10 ...
    ────────────────────────────────────────────────────────────────────────────────
    0    │ {vectors[0,0]:+.4f} {vectors[0,1]:+.4f} {vectors[0,2]:+.4f} {vectors[0,3]:+.4f} {vectors[0,4]:+.4f} {vectors[0,5]:+.4f} {vectors[0,6]:+.4f} {vectors[0,7]:+.4f} {vectors[0,8]:+.4f} {vectors[0,9]:+.4f} {vectors[0,10]:+.4f} ... (384 dims)
    1    │ {vectors[1,0]:+.4f} {vectors[1,1]:+.4f} {vectors[1,2]:+.4f} {vectors[1,3]:+.4f} {vectors[1,4]:+.4f} {vectors[1,5]:+.4f} {vectors[1,6]:+.4f} {vectors[1,7]:+.4f} {vectors[1,8]:+.4f} {vectors[1,9]:+.4f} {vectors[1,10]:+.4f} ...
    2    │ {vectors[2,0]:+.4f} {vectors[2,1]:+.4f} {vectors[2,2]:+.4f} {vectors[2,3]:+.4f} {vectors[2,4]:+.4f} {vectors[2,5]:+.4f} {vectors[2,6]:+.4f} {vectors[2,7]:+.4f} {vectors[2,8]:+.4f} {vectors[2,9]:+.4f} {vectors[2,10]:+.4f} ...
    3    │ {vectors[3,0]:+.4f} {vectors[3,1]:+.4f} {vectors[3,2]:+.4f} {vectors[3,3]:+.4f} {vectors[3,4]:+.4f} {vectors[3,5]:+.4f} {vectors[3,6]:+.4f} {vectors[3,7]:+.4f} {vectors[3,8]:+.4f} {vectors[3,9]:+.4f} {vectors[3,10]:+.4f} ...
    4    │ {vectors[4,0]:+.4f} {vectors[4,1]:+.4f} {vectors[4,2]:+.4f} {vectors[4,3]:+.4f} {vectors[4,4]:+.4f} {vectors[4,5]:+.4f} {vectors[4,6]:+.4f} {vectors[4,7]:+.4f} {vectors[4,8]:+.4f} {vectors[4,9]:+.4f} {vectors[4,10]:+.4f} ...
    """)
    
    # Matrice de similarité
    print("\n[Matrice de similarité entre chunks]")
    similarity_matrix = np.dot(vectors, vectors.T)
    
    print("\nSimilarité cosinus (produit scalaire de vecteurs normalisés):\n")
    print("     Chunk0 Chunk1 Chunk2 Chunk3 Chunk4")
    for i in range(5):
        row = f"Ch{i} │ "
        for j in range(5):
            val = similarity_matrix[i, j]
            bar_length = int(max(0, val) * 20)
            bar = "█" * bar_length
            row += f" {val:+.4f} {bar}  "
        print(row)
    
    print("\nDiagonale = 1.0 (chaque vecteur est identique à lui-même) ✓")


# ──────────────────────────────────────────────────────────────────────────────
# PARTIE 6 : COMPARAISON BM25 vs FAISS
# ──────────────────────────────────────────────────────────────────────────────

def demo_bm25_vs_faiss():
    """
    Montre la différence conceptuelle entre BM25 et FAISS.
    """
    print("\n" + "="*80)
    print("PARTIE 6 : COMPARAISON BM25 vs FAISS")
    print("="*80)
    
    print("""
┌──────────────────────┬──────────────────────────────────────────────┐
│ BM25 (Lexical)       │ FAISS (Sémantique)                           │
├──────────────────────┼──────────────────────────────────────────────┤
│ Basé sur: mots       │ Basé sur: sens du texte (embeddings)        │
│                      │                                              │
│ Recherche: mots clés │ Recherche: similarité sémantique            │
│                      │                                              │
│ Similarité: compte   │ Similarité: cosinus de vecteurs             │
│ des mots en commun   │ (tous les 384 dimensions considérées)      │
│                      │                                              │
│ Index: trie et stats │ Index: vecteurs en mémoire                  │
│ par terme            │ (structure FAISS = IndexFlatIP)             │
│                      │                                              │
│ Requête "résultats": │ Requête "résultats":                         │
│ cherche exactement   │ cherche le SENS "résultats"                 │
│ "résultats"         │ même si texte dit "findings" ou "conclusion" │
│                      │                                              │
│ Procédure:           │ Procédure:                                  │
│ 1. Tokenize "résultats"│ 1. Encode "résultats" → vecteur 384 dims  │
│ 2. Lookup dans BM25  │ 2. Calcul score = requête · chunk           │
│ 3. Return chunks     │ 3. Return top-K chunks par score            │
│    contenant ce mot  │    (même si mot différent)                  │
└──────────────────────┴──────────────────────────────────────────────┘

La fusion RRF combine les deux pour les avantages de chacun!
""")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "🎯 DÉMONSTRATION COMPLÈTE: VECTEURS ET SIMILARITÉ COSINUS" + "\n")
    
    # Part 1: Création embeddings
    emb1, emb2, txt1, txt2 = demo_embedding_creation()
    
    # Part 2: Stockage FAISS
    embeddings_list = [emb1, emb2, 
                       np.random.randn(384).astype(np.float32) / np.linalg.norm(np.random.randn(384)),
                       np.random.randn(384).astype(np.float32) / np.linalg.norm(np.random.randn(384)),
                       np.random.randn(384).astype(np.float32) / np.linalg.norm(np.random.randn(384))]
    
    # Normaliser tous les embeddings
    embeddings_list = [e / np.linalg.norm(e) for e in embeddings_list]
    
    index, embeddings_array = demo_faiss_storage(embeddings_list)
    
    # Part 3: Similarité cosinus
    sim = demo_cosine_similarity(emb1, emb2)
    
    # Part 4: Recherche requête
    demo_query_search(index, embeddings_array)
    
    # Part 5: Visualisation
    demo_vector_visualization()
    
    # Part 6: BM25 vs FAISS
    demo_bm25_vs_faiss()
    
    print("\n" + "="*80)
    print("📊 RÉSUMÉ FINAL")
    print("="*80)
    print("""
✓ Vecteurs créés : Texte → 384 nombres (float32) via SentenceTransformer
✓ Stockage : Tous les vecteurs en RAM, organisés par FAISS
✓ Indexation : O(n) pour ajouter, O(1) pour récupérer un vecteur
✓ Recherche : Calcul du score = requête · chunk (produit scalaire normalisé)
✓ Similarité : Cosinus entre [-1, 1] indique pertinence sémantique

Performance:
  - Créer embedding: 50 ms
  - Recherche 1000 chunks: 1-5 ms
  - Mémoire: 1000 chunks × 384 dims × 4 bytes ≈ 1.5 MB
  
Cas d'usage dans notre pipeline:
  1. Indexation (une seule fois):
     → Tous les chunks PDF → embeddings → FAISS storage
  
  2. Requête (chaque question):
     → Requête → embedding → chercher top-K dans FAISS
     → Combiner avec BM25 via RRF
     → Reranker pour meilleur score final
""")


if __name__ == "__main__":
    main()
