"""
pipeline_hf.py
==============
Pipeline RAG — 100% Hugging Face ONLINE & GRATUIT

✓ Zéro GPU requis
✓ Zéro API key requise  
✓ Zéro dépendance Ollama
✓ Validation automatique des modèles à chaque démarrage

Usage rapide:
  # 1. Indexation (une fois)
  python pipeline_hf.py --mode index --corpus ./corpus --index ./index
  
  # 2. Requêtes
  python pipeline_hf.py --mode run --questions questions.json --index ./index
  
  # 3. Pipeline complet
  python pipeline_hf.py --mode full --corpus ./corpus --questions questions.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from extractor import extraire_corpus
from indexer_hf_online import (
    IndexHybride,
    rechercher,
    test_modeles_startup,
    MODELE_EMBEDDING_DEFAUT,
    MODELE_RERANKER_DEFAUT,
    TOP_K_FINAL,
    TOP_K_INITIAL,
)
from generator_hf import (
    generer_reponse,
    formater_resultat_question,
    generer_soumission_json,
    sauvegarder_soumission,
)


# ══════════════════════════════════════════════════════════════════
# PHASES DU PIPELINE
# ══════════════════════════════════════════════════════════════════

def phase_indexation(dossier_corpus: Path,
                     dossier_index: Path,
                     utiliser_reranker: bool = True) -> IndexHybride:
    """Phase 1: Extraction PDF + construction index"""
    
    print("\n" + "="*70)
    print("📥 PHASE 1: EXTRACTION & INDEXATION")
    print("="*70)

    t0 = time.time()
    
    print(f"\n[Corpus] Lecture de {dossier_corpus}/")
    chunks = extraire_corpus(dossier_corpus)

    print(f"\n[Index] Construction hybride BM25 + FAISS...")
    index = IndexHybride(utiliser_reranker=utiliser_reranker)
    index.construire(chunks, batch_size=32)
    index.sauvegarder(dossier_index)

    dt = time.time() - t0
    print(f"\n✓ Indexation terminée en {dt:.1f}s")
    return index


def phase_run(index: IndexHybride,
              fichier_questions: Path,
              fichier_sortie: Path,
              top_k: int = TOP_K_FINAL,
              top_k_initial: int = TOP_K_INITIAL,
              mode_generation: str = "extraction") -> None:
    """Phase 2: Retrieval + génération"""
    
    print("\n" + "="*70)
    print("🔍 PHASE 2: RETRIEVAL & GÉNÉRATION")
    print("="*70)

    # Charger questions
    with open(fichier_questions, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    if "results" in questions_data:
        questions = questions_data["results"]
    elif isinstance(questions_data, list):
        questions = questions_data
    else:
        questions = [questions_data]

    print(f"\n[Questions] {len(questions)} à traiter")
    print(f"[Config] top_k={top_k}, mode_gen={mode_generation}")

    resultats = []
    t0 = time.time()

    for i, q in enumerate(questions):
        qid = q.get("qid", f"Q{i+1}")
        question = q.get("question", "")

        print(f"\n[{i+1}/{len(questions)}] {qid}: {question[:70]}...", flush=True)

        # ─ Retrieval
        t_r = time.time()
        chunks_recup = rechercher(index, question, top_k=top_k, top_k_initial=top_k_initial)
        dt_retrieval = time.time() - t_r

        # Afficher top-3
        for c in chunks_recup[:3]:
            print(f"    rank{c['rank']}: {c['doc_name']} p.{c['page']} "
                  f"[RRF:{c['score_rrf']:.4f}]")

        # ─ Génération
        t_g = time.time()
        reponse = generer_reponse(question, chunks_recup, mode=mode_generation)
        dt_gen = time.time() - t_g

        print(f"    Réponse ({dt_retrieval:.2f}s retrieval + {dt_gen:.2f}s gen):")
        print(f"      {reponse[:150]}...")

        # ─ Formatage
        resultat = formater_resultat_question(
            qid=qid,
            question=question,
            chunks_recuperes=chunks_recup,
            reponse=reponse,
        )
        resultats.append(resultat)

    # ─ Soumission JSON
    soumission = generer_soumission_json(
        resultats_questions=resultats,
        parametres_globaux={
            "top_k": top_k,
            "top_k_initial": top_k_initial,
            "mode_generation": mode_generation,
            "modele_embedding": MODELE_EMBEDDING_DEFAUT,
            "modele_reranker": MODELE_RERANKER_DEFAUT,
        }
    )
    sauvegarder_soumission(soumission, fichier_sortie)

    dt_total = time.time() - t0
    print(f"\n✓ {len(questions)} questions en {dt_total:.1f}s "
          f"({dt_total/len(questions):.2f}s/q)")


def creer_exemple_json(chemin: Path) -> None:
    """Crée fichier de questions d'exemple."""
    exemple = {
        "run_id": "questions-exemple",
        "parameters": {},
        "results": [
            {
                "qid": "Q1",
                "question": "Quelle est la méthodologie utilisée?",
                "retrieved": [],
                "answer": "",
                "metadata": {}
            },
            {
                "qid": "Q2",
                "question": "Quels sont les principaux résultats?",
                "retrieved": [],
                "answer": "",
                "metadata": {}
            },
        ]
    }
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(exemple, f, ensure_ascii=False, indent=2)
    print(f"✓ Exemple créé: {chemin}")


# ══════════════════════════════════════════════════════════════════
# CLI PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline RAG — 100% Hugging Face Online & Gratuit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--mode",
        choices=["index", "run", "full", "test", "demo"],
        default="demo",
        help="Mode d'exécution",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("corpus"),
        help="Dossier PDFs",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("index"),
        help="Dossier index",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("questions.json"),
        help="Fichier questions JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("soumission.json"),
        help="Fichier sortie JSON",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K_FINAL,
        help="Nb pages retournées",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Désactiver reranker",
    )
    parser.add_argument(
        "--generation",
        choices=["extraction", "llm-inference"],
        default="extraction",
        help="Mode génération",
    )

    args = parser.parse_args()

    # ── MODE TEST ────────────────────────────────────────────────────
    if args.mode == "test":
        print("\n" + "="*70)
        print("🧪 MODE TEST — Validation des modèles HF")
        print("="*70)
        ok = test_modeles_startup(
            embedding=MODELE_EMBEDDING_DEFAUT,
            reranker=MODELE_RERANKER_DEFAUT,
        )
        sys.exit(0 if ok else 1)

    # ── MODE DEMO ────────────────────────────────────────────────────
    if args.mode == "demo":
        print("\n" + "="*70)
        print("📚 MODE DÉMO — Pipeline 100% Hugging Face Online")
        print("="*70)
        print("""
Stack technique:
  ✓ Embedding: sentence-transformers/all-MiniLM-L6-v2 (22M, Apache 2.0)
  ✓ Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M, Apache 2.0)
  ✓ BM25: rank_bm25 (fournit lexical)
  ✓ FAISS: IndexFlatIP (fournit dense)
  ✓ RRF: fusion multi-ranking sans hyperparamètre

Étapes:
  1. python pipeline_hf.py --mode test
     → Valide que les modèles HF sont accessibles (~2 min)
  
  2. python pipeline_hf.py --mode index --corpus ./corpus --index ./index
     → Indexe les PDFs (utilise modèles HF, 1ère exécution te l'~5 min)
  
  3. python pipeline_hf.py --mode run --questions questions.json
     → Traite questions et génère soumission JSON

Ou directement:
  python pipeline_hf.py --mode full --corpus ./corpus --questions questions.json

Options:
  --top_k 10          : Nombre de pages par requête
  --no-rerank         : Mode rapide sans cross-encoder
  --generation extraction|llm-inference : Mode génération

Créer fichier questions d'exemple:
  python -c "from pipeline_hf import creer_exemple_json; \\
            creer_exemple_json(__file__)"
        """)
        return

    # ── MODE INDEX ───────────────────────────────────────────────────
    if args.mode in ("index", "full"):
        if not args.corpus.exists():
            print(f"✗ Dossier introuvable: {args.corpus}")
            sys.exit(1)
        
        # Test des modèles au démarrage
        if not test_modeles_startup():
            print("\n✗ Erreur: Modèles HF non accessibles")
            sys.exit(1)
        
        index = phase_indexation(
            dossier_corpus=args.corpus,
            dossier_index=args.index,
            utiliser_reranker=not args.no_rerank,
        )

    # ── MODE RUN ─────────────────────────────────────────────────────
    if args.mode in ("run", "full"):
        # Test des modèles au démarrage
        if not test_modeles_startup():
            print("\n✗ Erreur: Modèles HF non accessibles")
            sys.exit(1)
        
        # Si mode "run" seul, charger l'index depuis le disque
        if args.mode == "run":
            if not args.index.exists():
                print(f"✗ Index introuvable: {args.index}")
                print("  Lancez d'abord: --mode index")
                sys.exit(1)
            
            print(f"\n[Chargement] Index depuis {args.index}/...")
            index = IndexHybride(utiliser_reranker=not args.no_rerank)
            index.charger(args.index)

        if not args.questions.exists():
            print(f"✗ Questions introuvables: {args.questions}")
            print("  Créez d'abord un fichier JSON de questions")
            sys.exit(1)

        phase_run(
            index=index,
            fichier_questions=args.questions,
            fichier_sortie=args.output,
            top_k=args.top_k,
            mode_generation=args.generation,
        )


if __name__ == "__main__":
    main()
