"""
generator.py
============
Génération de réponses à partir des chunks récupérés.

Deux modes :
  1. Mode EXTRACTION (défaut, frugal) : répond directement depuis les chunks
     sans appel LLM externe — extrait et synthétise le texte brut.
     Recommandé pour tester rapidement et valider le retrieval.

  2. Mode LLM (complet) : utilise un LLM local via l'API Ollama (localhost:11434)
     ou tout endpoint compatible OpenAI. Modèle recommandé : Qwen2.5-7B-Instruct.

Format de sortie JSON conforme au challenge EvalLLM 2026.
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path


# ──────────────────────────────────────────────
# Paramètres par défaut
# ──────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODELE_LLM     = "qwen2.5:7b"           # ollama pull qwen2.5:7b
MAX_TOKENS_CTX = 3000                    # tokens max de contexte passés au LLM
RUN_ID         = "RAG-Challenge-EvalLLM2026"


# ──────────────────────────────────────────────
# Prompt de génération
# ──────────────────────────────────────────────
PROMPT_SYSTEME = """Tu es un assistant expert en recherche documentaire.
Tu réponds UNIQUEMENT en t'appuyant sur les extraits de documents fournis.
Si la réponse ne se trouve pas dans les extraits, dis-le explicitement.
Sois précis, complet et concis. Réponds en français."""

def construire_prompt(requete: str, chunks_recuperes: List[Dict]) -> str:
    """
    Construit le prompt LLM avec le contexte des chunks récupérés.
    Format : contexte numéroté + question.
    """
    contexte_parts = []
    for c in chunks_recuperes:
        header = (
            f"[Source {c['rank']}: {c['doc_name']}, "
            f"page {c['page']}, section: {c.get('section','N/A')}]"
        )
        contexte_parts.append(f"{header}\n{c['texte_brut']}")

    contexte = "\n\n---\n\n".join(contexte_parts)

    return f"""{PROMPT_SYSTEME}

EXTRAITS DE DOCUMENTS :
{contexte}

QUESTION : {requete}

RÉPONSE :"""


# ──────────────────────────────────────────────
# Mode extraction (sans LLM, frugal)
# ──────────────────────────────────────────────
def generer_reponse_extraction(
    requete: str,
    chunks_recuperes: List[Dict],
    nb_phrases: int = 5,
) -> str:
    """
    Génère une réponse par extraction directe sans LLM.

    Stratégie :
      1. Concatène les textes bruts des chunks récupérés.
      2. Score chaque phrase par overlap de mots avec la requête.
      3. Retourne les nb_phrases phrases les plus pertinentes, dans l'ordre original.

    Avantage : zéro GPU, zéro API, déterministe, très rapide.
    Limite : pas de reformulation ni de synthèse multi-sources.
    """
    if not chunks_recuperes:
        return "Aucun document pertinent trouvé pour cette requête."

    # Mots de la requête (sans stop words basiques)
    stop_words = {
        "le","la","les","de","du","des","un","une","et","en","à","au","aux",
        "est","sont","pour","que","qui","dans","par","sur","avec","il","elle",
        "ils","elles","on","se","si","mais","ou","donc","car","ni","or","the",
        "a","an","in","of","to","is","are","for","with","that","this","it"
    }
    mots_requete = {
        m.lower() for m in re.split(r'\W+', requete)
        if len(m) > 2 and m.lower() not in stop_words
    }

    # Extraire toutes les phrases des chunks (ordre source conservé)
    phrases_avec_source = []
    for chunk in chunks_recuperes:
        texte = chunk["texte_brut"]
        phrases = re.split(r'(?<=[.!?])\s+', texte)
        for phrase in phrases:
            phrase = phrase.strip()
            if len(phrase) > 30:  # ignorer les fragments très courts
                phrases_avec_source.append((phrase, chunk["rank"]))

    if not phrases_avec_source:
        return chunks_recuperes[0]["texte_brut"][:500] if chunks_recuperes else ""

    # Score par overlap de mots
    def score_phrase(phrase: str) -> float:
        mots_phrase = {
            m.lower() for m in re.split(r'\W+', phrase)
            if len(m) > 2 and m.lower() not in stop_words
        }
        if not mots_phrase:
            return 0.0
        overlap = len(mots_requete & mots_phrase)
        return overlap / (len(mots_phrase) ** 0.5)  # normalisation longueur

    phrases_scores = [
        (phrase, source, score_phrase(phrase))
        for phrase, source in phrases_avec_source
    ]

    # Top nb_phrases par score, mais conserver ordre original
    top_indices = sorted(
        range(len(phrases_scores)),
        key=lambda i: phrases_scores[i][2],
        reverse=True,
    )[:nb_phrases]
    top_indices_ordonnes = sorted(top_indices)

    reponse_phrases = [phrases_scores[i][0] for i in top_indices_ordonnes]
    return " ".join(reponse_phrases)


# ──────────────────────────────────────────────
# Mode LLM via Ollama
# ──────────────────────────────────────────────
def generer_reponse_llm(
    requete: str,
    chunks_recuperes: List[Dict],
    modele: str = MODELE_LLM,
    url: str = OLLAMA_URL,
    timeout: int = 120,
) -> str:
    """
    Génère une réponse via un LLM local (Ollama).

    Prérequis :
      - ollama installé et lancé : `ollama serve`
      - modèle disponible : `ollama pull qwen2.5:7b`

    Tronque le contexte si trop long (> MAX_TOKENS_CTX tokens approximatifs).
    """
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        return generer_reponse_extraction(requete, chunks_recuperes)

    # Tronquer les chunks si contexte trop long
    chunks_tronques = []
    tokens_total = 0
    for chunk in chunks_recuperes:
        tokens_chunk = max(1, len(chunk["texte_brut"]) // 4)
        if tokens_total + tokens_chunk > MAX_TOKENS_CTX:
            # Tronquer ce chunk
            reste = MAX_TOKENS_CTX - tokens_total
            if reste > 100:
                chunk_tronc = dict(chunk)
                chunk_tronc["texte_brut"] = chunk["texte_brut"][:reste * 4]
                chunks_tronques.append(chunk_tronc)
            break
        chunks_tronques.append(chunk)
        tokens_total += tokens_chunk

    prompt = construire_prompt(requete, chunks_tronques)

    payload = json.dumps({
        "model": modele,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,     # faible température = réponses factuelles
            "num_predict": 512,
            "top_p": 0.9,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        print(f"[LLM] Ollama non accessible ({e}). Fallback extraction.")
        return generer_reponse_extraction(requete, chunks_recuperes)
    except Exception as e:
        print(f"[LLM] Erreur : {e}. Fallback extraction.")
        return generer_reponse_extraction(requete, chunks_recuperes)


# ──────────────────────────────────────────────
# Générateur unifié
# ──────────────────────────────────────────────
def generer_reponse(
    requete: str,
    chunks_recuperes: List[Dict],
    mode: str = "extraction",  # "extraction" ou "llm"
    **kwargs,
) -> str:
    """
    Point d'entrée unifié pour la génération.

    mode="extraction" : rapide, sans GPU, sans API
    mode="llm"        : via Ollama local (Qwen2.5-7B recommandé)
    """
    if mode == "llm":
        return generer_reponse_llm(requete, chunks_recuperes, **kwargs)
    else:
        return generer_reponse_extraction(requete, chunks_recuperes, **kwargs)


# ──────────────────────────────────────────────
# Formatage JSON challenge EvalLLM 2026
# ──────────────────────────────────────────────
def formater_resultat_question(
    qid: str,
    question: str,
    chunks_recuperes: List[Dict],
    reponse: str,
    parametres: Optional[Dict] = None,
) -> Dict:
    """
    Formate un résultat selon le schéma JSON du challenge EvalLLM 2026.

    Format attendu :
    {
        "qid": "Q1",
        "question": "...",
        "retrieved": [
            {"rank": 1, "doc_name": "doc.pdf", "page": 3,
             "metadata": {...}}
        ],
        "answer": "...",
        "metadata": {...}
    }
    """
    retrieved = []
    for chunk in chunks_recuperes:
        retrieved.append({
            "rank"     : chunk["rank"],
            "doc_name" : chunk["doc_name"],
            "page"     : chunk["page"],
            "metadata" : {
                "section"        : chunk.get("section", ""),
                "score_rrf"      : chunk.get("score_rrf", 0.0),
                "score_reranker" : chunk.get("score_reranker", 0.0),
                "nb_pages_doc"   : chunk.get("nb_pages_doc", 0),
                "regime"         : chunk.get("regime", ""),
            }
        })

    return {
        "qid"      : qid,
        "question" : question,
        "retrieved": retrieved,
        "answer"   : reponse,
        "metadata" : {
            "parametres": parametres or {},
        }
    }


def generer_soumission_json(
    resultats_questions: List[Dict],
    run_id: str = RUN_ID,
    parametres_globaux: Optional[Dict] = None,
) -> Dict:
    """
    Génère la structure JSON complète de soumission au challenge.
    """
    return {
        "run_id"    : run_id,
        "parameters": parametres_globaux or {
            "modele_embedding" : "all-MiniLM-L6-v2",
            "modele_reranker"  : "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k_initial"    : 20,
            "top_k_final"      : 10,
            "rrf_k"            : 60,
            "overlap_tokens"   : 200,
            "seuil_section"    : 1.15,
        },
        "results"   : resultats_questions,
    }


def sauvegarder_soumission(soumission: Dict, chemin: Path) -> None:
    """Sauvegarde la soumission JSON."""
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(soumission, f, ensure_ascii=False, indent=2)
    print(f"[Output] Soumission sauvegardée : {chemin}")
    print(f"         {len(soumission['results'])} questions traitées.")


if __name__ == "__main__":
    print("generator.py chargé correctement.")
    print("Usage : from generator import generer_reponse, formater_resultat_question")
