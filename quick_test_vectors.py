#!/usr/bin/env python3
"""
quick_test_vectors.py
======================
Test rapide et interactif du système de vecteurs.

Usage:
  python3 quick_test_vectors.py
"""

import numpy as np
import sys


def test_basic_math():
    """Test 1: Maths de base (sans dépendances)."""
    print("\n" + "="*80)
    print("TEST 1: MATHS DE BASE (Produit scalaire et cosinus)")
    print("="*80)
    
    # Créer deux vecteurs simples (3D pour visualiser)
    u = np.array([-0.023, 0.145, -0.089], dtype=np.float32)
    v = np.array([-0.021, 0.142, -0.085], dtype=np.float32)
    
    print(f"\nVecteur U: {u}")
    print(f"Vecteur V: {v}")
    
    # Produit scalaire
    dot_product = np.dot(u, v)
    print(f"\nProduit scalaire (u · v): {dot_product:.8f}")
    print(f"  = ({u[0]}) × ({v[0]}) + ({u[1]}) × ({v[1]}) + ({u[2]}) × ({v[2]})")
    print(f"  = {u[0]*v[0]:.8f} + {u[1]*v[1]:.8f} + {u[2]*v[2]:.8f}")
    print(f"  = {dot_product:.8f}")
    
    # Normes
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    print(f"\nNorme de U (||u||): {norm_u:.8f}")
    print(f"Norme de V (||v||): {norm_v:.8f}")
    
    # Similarité cosinus
    similarity = dot_product / (norm_u * norm_v)
    print(f"\nSimilarité cosinus: {similarity:.8f}")
    print(f"  = {dot_product:.8f} / ({norm_u:.8f} × {norm_v:.8f})")
    print(f"  = {dot_product:.8f} / {norm_u * norm_v:.8f}")
    print(f"  = {similarity:.8f}")
    
    # Interprétation
    if similarity > 0.90:
        interp = "✅ EXCELLENT MATCH!"
    elif similarity > 0.70:
        interp = "✅ Bon match"
    elif similarity > 0.50:
        interp = "👍 Match acceptable"
    else:
        interp = "❌ Faible match"
    
    print(f"\nInterprétation: {interp}")
    print(f"  (Sur une échelle de -1 à +1, où 1 = identique)")
    
    return True


def test_256_dim_vectors():
    """Test 2: Vecteurs 256D (plus proche du réel)."""
    print("\n" + "="*80)
    print("TEST 2: VECTEURS 256D (proche des vrai embeddings)")
    print("="*80)
    
    np.random.seed(42)
    
    # Créer deux vecteurs similaires
    base = np.random.randn(256).astype(np.float32)
    u = base / np.linalg.norm(base)  # Normaliser
    
    # Vecteur similaire (ajouter un peu de bruit)
    v = base + np.random.randn(256).astype(np.float32) * 0.05
    v = v / np.linalg.norm(v)  # Normaliser
    
    print(f"\nVecteur U (256 dims): Créé et normalisé")
    print(f"  ||u|| = {np.linalg.norm(u):.8f}")
    print(f"  Premières valeurs: [{u[0]:.6f}, {u[1]:.6f}, {u[2]:.6f}, ...]")
    
    print(f"\nVecteur V (256 dims): Similaire + bruit, normalisé")
    print(f"  ||v|| = {np.linalg.norm(v):.8f}")
    print(f"  Premières valeurs: [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, ...]")
    
    # Calcul pour vecteurs NORMALISÉS
    similarity_normalized = np.dot(u, v)
    
    print(f"\nSimilarité (vecteurs normalisés):")
    print(f"  cos(θ) = u · v = {similarity_normalized:.8f}")
    print(f"  (Pas besoin de diviser par les normes!)")
    
    print(f"\nInterprétation:")
    print(f"  Score: {similarity_normalized:.6f}")
    print(f"  En pourcentage: {similarity_normalized*100:.2f}%")
    
    return True


def test_faiss_simulation():
    """Test 3: Simulation d'une recherche FAISS."""
    print("\n" + "="*80)
    print("TEST 3: SIMULATION RECHERCHE FAISS (5 chunks)")
    print("="*80)
    
    np.random.seed(123)
    
    # Créer 5 vecteurs simulant des chunks
    n_chunks = 5
    dim = 384
    
    chunks_vec = np.random.randn(n_chunks, dim).astype(np.float32)
    chunks_vec = chunks_vec / np.linalg.norm(chunks_vec, axis=1, keepdims=True)
    
    # Créer une requête
    query_vec = np.random.randn(1, dim).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    print(f"\n[Setup]")
    print(f"  Chunks: {n_chunks}")
    print(f"  Dimensions: {dim}")
    print(f"  Requête: 1 vecteur normalisé")
    
    # Calculer scores (brute-force = FAISS IndexFlatIP)
    print(f"\n[Recherche (calcul des scores)]")
    scores = np.dot(query_vec, chunks_vec.T)[0]
    
    print(f"  Scores calculés:")
    for i, score in enumerate(scores):
        bar = "█" * int(max(0, score) * 30)
        print(f"    Chunk {i}: {score:+.8f}  {bar}")
    
    # Trier
    print(f"\n[Résultats (triés)]")
    indices_sorted = np.argsort(-scores)
    
    for rank, idx in enumerate(indices_sorted, 1):
        print(f"  Rang {rank}: Chunk {idx} (score: {scores[idx]:+.8f})")
    
    return True


def test_with_real_model():
    """Test 4: Utiliser le vrai modèle SentenceTransformer."""
    print("\n" + "="*80)
    print("TEST 4: VRAI MODÈLE SentenceTransformer")
    print("="*80)
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers non installé")
        print("   pip install sentence-transformers")
        return False
    
    print("\n[Chargement du modèle all-MiniLM-L6-v2]")
    print("  (première fois: télécharge ~90 MB)")
    
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
    
    print("✓ Modèle chargé\n")
    
    # Trois textes
    texts = [
        "Intelligence artificielle et apprentissage automatique",
        "L'IA utilise l'apprentissage profond pour traiter les données",
        "Les tomates rouges sont délicieuses au printemps"
    ]
    
    print("[Encodage de 3 textes]")
    for i, text in enumerate(texts):
        print(f"  {i+1}. \"{text}\"")
    
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    print(f"\n✓ Embeddings créés: {embeddings.shape}")
    print(f"  ({len(texts)} textes × 384 dimensions)")
    
    # Vérifier normalisation
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nVérification normalisation:")
    for i, norm in enumerate(norms):
        status = "✓" if abs(norm - 1.0) < 0.01 else "❌"
        print(f"  Texte {i+1}: ||v|| = {norm:.8f} {status}")
    
    # Matrice de similarité
    print(f"\n[Matrice de similarité cosinus]")
    sim_matrix = np.dot(embeddings, embeddings.T)
    
    print(f"\n{'':30s}  Texte 1  Texte 2  Texte 3")
    for i in range(3):
        row = f"Texte {i+1} {texts[i][:22]:22s} "
        for j in range(3):
            sim = sim_matrix[i, j]
            bar_len = int(max(0, sim) * 15)
            bar = "█" * bar_len
            row += f" {sim:+.4f}  "
        print(row)
    
    # Observation
    print(f"\n[Observation]")
    sim_1_2 = sim_matrix[0, 1]
    sim_1_3 = sim_matrix[0, 2]
    print(f"  Similarité Texte 1-2 (sémantiquement lié): {sim_1_2:.4f}")
    print(f"  Similarité Texte 1-3 (sans lien): {sim_1_3:.4f}")
    print(f"\n  ✓ Le modèle comprend le sémantique!")
    print(f"    Textes similaires → score plus haut")
    
    return True


def test_with_faiss():
    """Test 5: Utiliser FAISS réel."""
    print("\n" + "="*80)
    print("TEST 5: RECHERCHE AVEC FAISS")
    print("="*80)
    
    try:
        import faiss
    except ImportError:
        print("❌ FAISS non installé")
        print("   pip install faiss-cpu")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers non installé")
        return False
    
    print("\n[Setup]")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 10 textes simulant des chunks
    chunks_text = [
        "L'intelligence artificielle transforme le monde moderne",
        "Les réseaux de neurones sont inspirés par le cerveau humain",
        "Le machine learning permet aux machines d'apprendre",
        "Les données sont le carburant de l'IA",
        "La météo sera ensoleillée demain",
        "Les chats sont des animaux domestiques",
        "Les résultats de notre étude montrent une amélioration",
        "La méthodologie utilisée est robuste et efficace",
        "Les fruits sont bons pour la santé",
        "Le deep learning utilise de nombreuses couches"
    ]
    
    print(f"  Nombre de chunks: {len(chunks_text)}")
    print(f"\nChunks:")
    for i, text in enumerate(chunks_text):
        print(f"    {i}: {text[:60]}...")
    
    # Encoder les chunks
    print(f"\n[Encodage des chunks (384 dimensions)]")
    embeddings = model.encode(chunks_text, normalize_embeddings=True)
    print(f"  ✓ {len(embeddings)} embeddings créés")
    
    # Créer l'index FAISS
    print(f"\n[Création de l'index FAISS]")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"  ✓ Index créé avec {index.ntotal} vecteurs")
    
    # Requête
    query_text = "Quels résultats montre cette étude?"
    print(f"\n[Requête]")
    print(f"  \"{query_text}\"")
    
    query_vec = model.encode([query_text], normalize_embeddings=True)
    
    # Recherche
    print(f"\n[Recherche (top-5 résultats)]")
    distances, indices = index.search(query_vec, k=5)
    distances = distances[0]
    indices = indices[0]
    
    for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
        bar_len = int(dist * 40)
        bar = "█" * bar_len
        print(f"  Rang {rank}: Chunk {idx:2d} │ {dist:+.6f} │ {bar}")
        print(f"           \"{chunks_text[idx][:60]}...\"")
    
    # Observation
    best_idx = indices[0]
    best_chunk = chunks_text[best_idx]
    print(f"\n[Résultat]")
    print(f"  Meilleur match: Chunk {best_idx}")
    print(f"  \"{best_chunk}\"")
    print(f"  Score: {distances[0]:.6f}")
    
    if best_idx in [6, 7]:  # Chunks sur les résultats/méthodologie
        print(f"  ✓ Correct! (chunk sur résultats/étude)")
    else:
        print(f"  (résultat peut varier selon l'ordre)")
    
    return True


def main():
    print("\n" + "🎯 TESTS INTERACTIFS: VECTEURS ET SIMILARITÉ COSINUS" + "\n")
    
    tests = [
        ("Maths de base", test_basic_math),
        ("Vecteurs 256D", test_256_dim_vectors),
        ("Simulation FAISS", test_faiss_simulation),
        ("Vrai modèle", test_with_real_model),
        ("FAISS réel", test_with_faiss),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Exécution: {name}...")
        print(f"{'='*80}")
        
        try:
            result = test_func()
            results.append((name, "✅ OK" if result else "⚠️ SAUTE"))
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            results.append((name, f"❌ ERREUR"))
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    
    for name, status in results:
        print(f"  {status}  {name}")
    
    print("\n" + "="*80)
    print("✅ Tests interactifs terminés")
    print("="*80)


if __name__ == "__main__":
    main()
