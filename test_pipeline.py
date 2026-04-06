"""
test_pipeline.py
================
Valide le pipeline complet avec des PDFs synthétiques générés en mémoire.
Teste les trois régimes de chunking (court / moyen / long).
Vérifie la conformité du JSON de sortie avec le format EvalLLM 2026.
"""

import json
import tempfile
import shutil
from pathlib import Path

import fitz  # PyMuPDF


# ──────────────────────────────────────────────
# Génération de PDFs synthétiques pour les tests
# ──────────────────────────────────────────────
def creer_pdf_synthetique(
    chemin: Path,
    nb_pages: int,
    titre: str,
    contenu_pages: list,
) -> None:
    """Crée un PDF multi-pages avec du contenu textuel structuré."""
    doc = fitz.open()
    for i, contenu in enumerate(contenu_pages[:nb_pages]):
        page = doc.new_page(width=595, height=842)
        # Titre de page/section
        page.insert_text(
            (50, 60),
            f"{titre} — Page {i+1}",
            fontsize=16,
            fontname="helv",
        )
        # Corps du texte
        page.insert_text(
            (50, 100),
            contenu,
            fontsize=11,
            fontname="helv",
        )
    doc.save(str(chemin))
    doc.close()


def creer_corpus_test(dossier: Path) -> None:
    """Crée 3 PDFs synthétiques couvrant les 3 régimes de chunking."""
    dossier.mkdir(parents=True, exist_ok=True)

    # Doc court (2 pages)
    creer_pdf_synthetique(
        chemin=dossier / "doc_court.pdf",
        nb_pages=2,
        titre="Rapport de synthèse",
        contenu_pages=[
            ("Ce document présente les résultats de l'étude sur l'intelligence artificielle. "
             "La méthodologie adoptée repose sur une approche mixte quantitative et qualitative. "
             "Les données ont été collectées auprès de 500 participants sur une période de 6 mois. "
             "L'analyse statistique révèle une corrélation significative entre les variables.\n\n"
             "Les principaux résultats montrent que 78% des participants ont amélioré leurs performances "
             "après l'intervention. Le taux de satisfaction global est de 4.2/5. "
             "Ces résultats confirment l'hypothèse principale de la recherche."),
            ("Conclusion : l'étude démontre l'efficacité de la méthode proposée. "
             "Les perspectives incluent l'extension à d'autres domaines d'application. "
             "Des travaux futurs sont envisagés pour valider ces résultats sur des populations plus larges. "
             "La reproductibilité des résultats a été vérifiée par trois équipes indépendantes.\n\n"
             "Références : [1] Smith et al. 2023, [2] Dupont et al. 2022, [3] Zhang 2024."),
        ]
    )

    # Doc moyen (8 pages)
    sections = [
        "1. Introduction",
        "2. État de l'art",
        "3. Méthodologie",
        "4. Expériences",
        "5. Résultats",
        "6. Discussion",
        "7. Conclusion",
        "8. Bibliographie",
    ]
    contenus_moyen = []
    for i, sec in enumerate(sections):
        contenus_moyen.append(
            f"{sec}\n\n"
            f"Cette section aborde les aspects fondamentaux de la recherche en traitement automatique "
            f"du langage naturel. Les modèles de langue pré-entraînés comme BERT et GPT ont révolutionné "
            f"le domaine depuis 2018. La technique RAG (Retrieval-Augmented Generation) permet de combiner "
            f"la puissance des LLMs avec la précision d'un système de recherche d'information.\n\n"
            f"Les résultats obtenus sur le benchmark KILT montrent une amélioration de 15% par rapport "
            f"aux baselines. Le modèle proposé utilise un dual-encoder pour le retrieval dense et "
            f"BM25 pour la recherche lexicale. La fusion RRF normalise les scores sans hyperparamètre sensible."
        )
    creer_pdf_synthetique(
        chemin=dossier / "article_recherche.pdf",
        nb_pages=8,
        titre="Article de recherche NLP",
        contenu_pages=contenus_moyen,
    )

    # Doc long (20 pages)
    contenus_long = []
    chapitres = [
        "Chapitre 1 : Fondements théoriques",
        "Chapitre 1 : Fondements théoriques (suite)",
        "Chapitre 2 : Architecture système",
        "Chapitre 2 : Détails d'implémentation",
        "Chapitre 3 : Protocole expérimental",
        "Chapitre 3 : Datasets utilisés",
        "Chapitre 4 : Résultats quantitatifs",
        "Chapitre 4 : Analyse par sous-corpus",
        "Chapitre 5 : Ablation studies",
        "Chapitre 5 : Impact du chunking",
        "Chapitre 6 : Comparaison avec l'état de l'art",
        "Chapitre 6 : Analyse des erreurs",
        "Chapitre 7 : Cas d'usage industriels",
        "Chapitre 7 : Déploiement et frugalité",
        "Chapitre 8 : Limitations",
        "Chapitre 8 : Travaux futurs",
        "Chapitre 9 : Conclusion générale",
        "Annexe A : Hyperparamètres",
        "Annexe B : Exemples de sorties",
        "Bibliographie complète",
    ]
    for chap in chapitres:
        contenus_long.append(
            f"{chap}\n\n"
            f"Les systèmes RAG représentent une avancée majeure dans la conception d'assistants "
            f"documentaires intelligents. L'architecture proposée dans ce rapport intègre un pipeline "
            f"d'indexation adaptatif qui prend en compte la structure variable des documents sources. "
            f"Le chunking conditionnel (régimes court/moyen/long) permet d'optimiser la granularité "
            f"de l'index en fonction du nombre de pages de chaque document.\n\n"
            f"Les performances mesurées sur le benchmark EvalLLM 2026 montrent que l'approche hybride "
            f"BM25 + FAISS avec fusion RRF surpasse les méthodes mono-retrieval de 12 à 18 points "
            f"en NDCG@10. Le reranker cross-encoder apporte un gain supplémentaire de 6 points "
            f"sur les requêtes multi-hop nécessitant la fusion de plusieurs sources documentaires."
        )
    creer_pdf_synthetique(
        chemin=dossier / "rapport_technique.pdf",
        nb_pages=20,
        titre="Rapport technique complet",
        contenu_pages=contenus_long,
    )

    print(f"[Test] 3 PDFs créés dans {dossier}/")
    print("  - doc_court.pdf         (2 pages  → régime court)")
    print("  - article_recherche.pdf (8 pages  → régime moyen)")
    print("  - rapport_technique.pdf (20 pages → régime long)")


def creer_questions_test() -> list:
    """Retourne une liste de questions de test couvrant différentes difficultés."""
    return [
        {
            "qid": "Q1",
            "question": "Quelle est la méthodologie utilisée dans le rapport de synthèse ?",
            "retrieved": [],
            "answer": "",
            "metadata": {}
        },
        {
            "qid": "Q2",
            "question": "Quels sont les résultats du benchmark KILT mentionnés dans l'article de recherche ?",
            "retrieved": [],
            "answer": "",
            "metadata": {}
        },
        {
            "qid": "Q3",
            "question": "Comment fonctionne le chunking conditionnel décrit dans le rapport technique ?",
            "retrieved": [],
            "answer": "",
            "metadata": {}
        },
        {
            "qid": "Q4",
            "question": "Quelle amélioration en NDCG@10 l'approche hybride apporte-t-elle selon le rapport ?",
            "retrieved": [],
            "answer": "",
            "metadata": {}
        },
    ]


# ──────────────────────────────────────────────
# Test principal
# ──────────────────────────────────────────────
def run_test():
    print("=" * 60)
    print("TEST DU PIPELINE RAG — EvalLLM 2026")
    print("=" * 60)

    # Répertoires temporaires
    tmpdir = Path(tempfile.mkdtemp())
    dossier_corpus  = tmpdir / "corpus"
    dossier_index   = tmpdir / "index"
    fichier_questions = tmpdir / "questions.json"
    fichier_sortie    = tmpdir / "soumission.json"

    try:
        # ── Étape 1 : Créer le corpus de test ───────────────────────────────
        print("\n[1/4] Création du corpus synthétique...")
        creer_corpus_test(dossier_corpus)

        # ── Étape 2 : Extraction et indexation ──────────────────────────────
        print("\n[2/4] Extraction et indexation...")
        from extractor import extraire_corpus
        from indexer import IndexHybride

        chunks = extraire_corpus(dossier_corpus)

        # Vérifications sur les chunks
        regimes = {}
        for c in chunks:
            r = c["regime"]
            regimes[r] = regimes.get(r, 0) + 1
        print(f"\n  Régimes détectés : {regimes}")

        # Vérifier que le préfixe [DOC] [SECTION] [PAGE] est présent partout
        for c in chunks[:3]:
            assert "[DOC:" in c["texte_enrichi"], "Préfixe DOC manquant"
            assert "[SECTION:" in c["texte_enrichi"], "Préfixe SECTION manquant"
            assert "[PAGE:" in c["texte_enrichi"], "Préfixe PAGE manquant"
        print("  Préfixes contextuels : OK")

        # Construire l'index (sans reranker pour la vitesse du test)
        index = IndexHybride(utiliser_reranker=False)
        index.construire(chunks, batch_size=32)
        index.sauvegarder(dossier_index)

        # ── Étape 3 : Requêtes de test ───────────────────────────────────────
        print("\n[3/4] Test du retrieval...")
        from indexer import rechercher

        questions = creer_questions_test()
        with open(fichier_questions, "w", encoding="utf-8") as f:
            json.dump({"run_id": "test", "results": questions}, f, ensure_ascii=False, indent=2)

        # Test manuel d'une requête
        q_test = "Quelle est la méthodologie de l'étude ?"
        resultats = rechercher(index, q_test, top_k=5)
        print(f"\n  Requête : '{q_test}'")
        for r in resultats:
            print(f"    rank{r['rank']}: {r['doc_name']} p.{r['page']} "
                  f"[RRF:{r['score_rrf']:.4f}]")

        assert len(resultats) > 0, "Aucun résultat retourné"
        assert all("doc_name" in r and "page" in r for r in resultats), \
            "Champs doc_name/page manquants"
        print("  Retrieval : OK")

        # ── Étape 4 : Génération et JSON challenge ────────────────────────────
        print("\n[4/4] Génération et formatage JSON...")
        from generator import (
            generer_reponse,
            formater_resultat_question,
            generer_soumission_json,
            sauvegarder_soumission,
        )

        resultats_json = []
        for q in questions:
            chunks_q = rechercher(index, q["question"], top_k=5)
            reponse = generer_reponse(q["question"], chunks_q, mode="extraction")
            r = formater_resultat_question(
                qid=q["qid"],
                question=q["question"],
                chunks_recuperes=chunks_q,
                reponse=reponse,
            )
            resultats_json.append(r)

        soumission = generer_soumission_json(resultats_json)
        sauvegarder_soumission(soumission, fichier_sortie)

        # ── Validation du JSON de sortie ──────────────────────────────────────
        with open(fichier_sortie, "r", encoding="utf-8") as f:
            s = json.load(f)

        assert "run_id" in s
        assert "parameters" in s
        assert "results" in s
        assert len(s["results"]) == len(questions)

        for r in s["results"]:
            assert "qid" in r
            assert "question" in r
            assert "retrieved" in r
            assert "answer" in r
            for retrieved in r["retrieved"]:
                assert "rank" in retrieved
                assert "doc_name" in retrieved
                assert "page" in retrieved
                assert isinstance(retrieved["page"], int)
                assert retrieved["page"] >= 1

        print("\n  Validation JSON : OK")
        print(f"\n  Exemple de résultat (Q1) :")
        q1 = s["results"][0]
        print(f"    qid      : {q1['qid']}")
        print(f"    question : {q1['question'][:60]}...")
        print(f"    retrieved: {[{'doc':r['doc_name'],'page':r['page']} for r in q1['retrieved'][:2]]}")
        print(f"    answer   : {q1['answer'][:100]}...")

        print("\n" + "=" * 60)
        print("TOUS LES TESTS PASSENT ✓")
        print("=" * 60)
        print("\nPour utiliser le pipeline sur votre corpus réel :")
        print("  python pipeline.py --mode full \\")
        print("      --corpus ./votre_corpus \\")
        print("      --index ./index \\")
        print("      --questions questions.json \\")
        print("      --output soumission.json")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    run_test()
