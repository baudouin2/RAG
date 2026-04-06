"""
pipeline.py
===========
Point d'entrée principal du pipeline RAG pour le challenge EvalLLM 2026.

Usage :
  # 1. Indexation (une seule fois sur le corpus)
  python pipeline.py --mode index --corpus ./corpus --index ./index

  # 2. Réponse à un fichier de questions (format JSON challenge)
  python pipeline.py --mode run \
      --questions questions.json \
      --index ./index \
      --output soumission.json \
      --generation extraction   # ou "llm" si Ollama disponible

  # 3. Pipeline complet (index + run en une commande)
  python pipeline.py --mode full \
      --corpus ./corpus \
      --index ./index \
      --questions questions.json \
      --output soumission.json

Options :
  --top_k     : nombre de pages retournées par requête (défaut: 10)
  --no-rerank : désactiver le reranker (plus rapide, moins précis)
  --generation: "extraction" (défaut) ou "llm" (nécessite Ollama)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from extractor import extraire_corpus
from indexer import IndexHybride, rechercher, TOP_K_FINAL, TOP_K_INITIAL
from generator import (
    generer_reponse,
    formater_resultat_question,
    generer_soumission_json,
    sauvegarder_soumission,
    RUN_ID,
)


# ──────────────────────────────────────────────
# Phase 1 : Indexation
# ──────────────────────────────────────────────
def phase_indexation(
    dossier_corpus: Path,
    dossier_index: Path,
    utiliser_reranker: bool = True,
) -> IndexHybride:
    """
    Extrait tous les PDFs du corpus, construit l'index hybride, sauvegarde.
    """
    print("=" * 60)
    print("PHASE 1 : EXTRACTION ET INDEXATION")
    print("=" * 60)

    t0 = time.time()
    print(f"\n[Corpus] Lecture de {dossier_corpus}/")
    chunks = extraire_corpus(dossier_corpus)

    print(f"\n[Index] Construction de l'index hybride...")
    index = IndexHybride(utiliser_reranker=utiliser_reranker)
    index.construire(chunks)
    index.sauvegarder(dossier_index)

    dt = time.time() - t0
    print(f"\n[OK] Indexation terminée en {dt:.1f}s")
    return index


# ──────────────────────────────────────────────
# Phase 2 : Réponses aux questions
# ──────────────────────────────────────────────
def phase_run(
    index: IndexHybride,
    fichier_questions: Path,
    fichier_sortie: Path,
    top_k: int = TOP_K_FINAL,
    top_k_initial: int = TOP_K_INITIAL,
    mode_generation: str = "extraction",
) -> None:
    """
    Traite toutes les questions du fichier JSON et génère la soumission.

    Le fichier de questions suit le format challenge (champs retrieved et answer vides).
    """
    print("\n" + "=" * 60)
    print("PHASE 2 : RETRIEVAL ET GÉNÉRATION")
    print("=" * 60)

    # Chargement des questions
    with open(fichier_questions, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    # Le format challenge encapsule les questions dans "results"
    if "results" in questions_data:
        questions = questions_data["results"]
    elif isinstance(questions_data, list):
        questions = questions_data
    else:
        # Format minimal : liste de {qid, question}
        questions = [questions_data]

    print(f"\n[Questions] {len(questions)} questions à traiter.")
    print(f"[Config] top_k={top_k}, generation={mode_generation}")

    resultats = []
    t0 = time.time()

    for i, q in enumerate(questions):
        qid      = q.get("qid", f"Q{i+1}")
        question = q.get("question", "")

        print(f"\n[{i+1}/{len(questions)}] {qid}: {question[:80]}...")

        # ── Retrieval ───────────────────────────────────────────────────────
        t_r = time.time()
        chunks_recuperes = rechercher(
            index,
            question,
            top_k=top_k,
            top_k_initial=top_k_initial,
        )
        dt_retrieval = time.time() - t_r

        # Afficher les top-3 résultats
        for c in chunks_recuperes[:3]:
            print(f"    rank{c['rank']}: {c['doc_name']} p.{c['page']} "
                  f"[RRF:{c['score_rrf']:.4f} RE:{c['score_reranker']:.3f}]")

        # ── Génération ──────────────────────────────────────────────────────
        t_g = time.time()
        reponse = generer_reponse(question, chunks_recuperes, mode=mode_generation)
        dt_gen = time.time() - t_g

        print(f"    Réponse ({dt_retrieval:.2f}s retrieval + {dt_gen:.2f}s gen): "
              f"{reponse[:120]}...")

        # ── Formatage ───────────────────────────────────────────────────────
        resultat = formater_resultat_question(
            qid=qid,
            question=question,
            chunks_recuperes=chunks_recuperes,
            reponse=reponse,
        )
        resultats.append(resultat)

    # ── Soumission JSON ──────────────────────────────────────────────────────
    soumission = generer_soumission_json(
        resultats_questions=resultats,
        run_id=RUN_ID,
        parametres_globaux={
            "top_k"            : top_k,
            "top_k_initial"    : top_k_initial,
            "mode_generation"  : mode_generation,
            "modele_embedding" : "all-MiniLM-L6-v2",
            "modele_reranker"  : "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rrf_k"            : 60,
            "overlap_tokens"   : 200,
            "seuil_titre_ratio": 1.15,
        }
    )
    sauvegarder_soumission(soumission, fichier_sortie)

    dt_total = time.time() - t0
    print(f"\n[OK] {len(questions)} questions traitées en {dt_total:.1f}s "
          f"({dt_total/len(questions):.2f}s/question)")


# ──────────────────────────────────────────────
# Utilitaire : créer un fichier de questions d'exemple
# ──────────────────────────────────────────────
def creer_questions_exemple(chemin: Path) -> None:
    """Crée un fichier de questions dans le format du challenge pour les tests."""
    exemple = {
        "run_id": "questions-exemple",
        "parameters": {},
        "results": [
            {
                "qid": "Q1",
                "question": "Quelle est la méthodologie utilisée dans ce document ?",
                "retrieved": [],
                "answer": "",
                "metadata": {}
            },
            {
                "qid": "Q2",
                "question": "Quels sont les principaux résultats obtenus ?",
                "retrieved": [],
                "answer": "",
                "metadata": {}
            },
            {
                "qid": "Q3",
                "question": "Quelles sont les conclusions et perspectives de ce travail ?",
                "retrieved": [],
                "answer": "",
                "metadata": {}
            }
        ]
    }
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(exemple, f, ensure_ascii=False, indent=2)
    print(f"[Exemple] Fichier de questions créé : {chemin}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline RAG hybride pour le challenge EvalLLM 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["index", "run", "full", "demo"],
        default="demo",
        help="Mode d'exécution",
    )
    parser.add_argument("--corpus",    type=Path, default=Path("corpus"),
                        help="Dossier contenant les PDFs")
    parser.add_argument("--index",     type=Path, default=Path("index"),
                        help="Dossier de sauvegarde de l'index")
    parser.add_argument("--questions", type=Path, default=Path("questions.json"),
                        help="Fichier JSON des questions (format challenge)")
    parser.add_argument("--output",    type=Path, default=Path("soumission.json"),
                        help="Fichier JSON de sortie (soumission challenge)")
    parser.add_argument("--top_k",     type=int,  default=TOP_K_FINAL,
                        help="Nombre de pages retournées par requête")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Désactiver le reranker")
    parser.add_argument("--generation", choices=["extraction", "llm"],
                        default="extraction",
                        help="Mode de génération des réponses")

    args = parser.parse_args()

    # ── Mode DEMO ────────────────────────────────────────────────────────────
    if args.mode == "demo":
        print("=" * 60)
        print("MODE DÉMO — Validation du pipeline sans PDFs réels")
        print("=" * 60)
        print("\nPour utiliser le pipeline :")
        print("  1. Placer vos PDFs dans ./corpus/")
        print("  2. python pipeline.py --mode index --corpus ./corpus --index ./index")
        print("  3. python pipeline.py --mode run \\")
        print("         --questions questions.json \\")
        print("         --index ./index \\")
        print("         --output soumission.json")
        print("\nPour créer un fichier de questions d'exemple :")
        print("  python -c \"from pipeline import creer_questions_exemple; "
              "from pathlib import Path; creer_questions_exemple(Path('questions_exemple.json'))\"")
        print("\nStack technique :")
        print("  - Extraction  : PyMuPDF")
        print("  - Embedding   : all-MiniLM-L6-v2 (22M params)")
        print("  - Index dense : FAISS IndexFlatIP")
        print("  - Index sparse: BM25Okapi (rank_bm25)")
        print("  - Fusion      : Reciprocal Rank Fusion (k=60)")
        print("  - Reranker    : cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params)")
        print("  - Génération  : extraction (sans GPU) ou Ollama/Qwen2.5-7B")
        return

    # ── Mode INDEX ───────────────────────────────────────────────────────────
    if args.mode in ("index", "full"):
        if not args.corpus.exists():
            print(f"[Erreur] Dossier corpus introuvable : {args.corpus}")
            sys.exit(1)
        index = phase_indexation(
            dossier_corpus=args.corpus,
            dossier_index=args.index,
            utiliser_reranker=not args.no_rerank,
        )

    # ── Mode RUN ─────────────────────────────────────────────────────────────
    if args.mode in ("run", "full"):
        # Si mode "run", charger l'index depuis le disque
        if args.mode == "run":
            if not args.index.exists():
                print(f"[Erreur] Index introuvable : {args.index}")
                print("  Lancez d'abord : python pipeline.py --mode index")
                sys.exit(1)
            index = IndexHybride(utiliser_reranker=not args.no_rerank)
            index.charger(args.index)

        if not args.questions.exists():
            print(f"[Erreur] Fichier de questions introuvable : {args.questions}")
            print("  Créer un exemple : from pipeline import creer_questions_exemple")
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
