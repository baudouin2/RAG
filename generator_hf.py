"""
generator_hf.py
==============
Génération de réponses — 100% Hugging Face ONLINE & GRATUIT

Deux modes:
  1. Mode EXTRACTION (défaut, ultra-frugal): extraction + synthèse texte
  2. Mode LLM (optionnel): utilise inférence.ai ou autre LLM public

Zéro API key requise, zéro GPU nécessaire.
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
# MODE EXTRACTION (défaut)
# ══════════════════════════════════════════════════════════════════

def generer_reponse_extraction(
    requete: str,
    chunks_recuperes: List[Dict],
    nb_phrases: int = 5,
) -> str:
    """
    Génère réponse par extraction directe (zéro LLM, ultra-rapide).
    
    Stratégie:
      1. Concatène textes des chunks
      2. Score chaque phrase par overlap de mots avec requête
      3. Retourne les TOP N phrases dans l'ordre original
    """
    if not chunks_recuperes:
        return "Aucun document pertinent trouvé pour cette requête."

    # Stop words basiques (français + anglais)
    stop_words = {
        "le","la","les","de","du","des","un","une","et","en","à","au","aux",
        "est","sont","pour","que","qui","dans","par","sur","avec","il","elle",
        "ils","elles","on","se","si","mais","ou","donc","car","ni","or","the",
        "a","an","in","of","to","is","are","for","with","that","this","it"
    }
    
    # Extraire mots clés de la requête
    mots_requete = {
        m.lower() for m in re.split(r'\W+', requete)
        if len(m) > 2 and m.lower() not in stop_words
    }

    # Extraire phrases des chunks (ordre source conservé)
    phrases_avec_source = []
    for chunk in chunks_recuperes:
        texte = chunk.get("texte_brut", "")
        phrases = re.split(r'(?<=[.!?])\s+', texte)
        for phrase in phrases:
            phrase = phrase.strip()
            if len(phrase) > 30:  # ignorer fragments trop courts
                phrases_avec_source.append((phrase, chunk["rank"]))

    if not phrases_avec_source:
        return chunks_recuperes[0].get("texte_brut", "")[:500]

    # Score chaque phrase par overlap de mots
    def score_phrase(phrase: str) -> float:
        mots_phrase = {
            m.lower() for m in re.split(r'\W+', phrase)
            if len(m) > 2 and m.lower() not in stop_words
        }
        if not mots_phrase:
            return 0.0
        overlap = len(mots_requete & mots_phrase)
        return overlap / (len(mots_phrase) ** 0.5)  # normalisation

    phrases_scores = [
        (phrase, source, score_phrase(phrase))
        for phrase, source in phrases_avec_source
    ]

    # Top N phrases, ordre original
    top_indices = sorted(
        range(len(phrases_scores)),
        key=lambda i: phrases_scores[i][2],
        reverse=True,
    )[:nb_phrases]
    top_indices_ordonnes = sorted(top_indices)

    reponse_phrases = [phrases_scores[i][0] for i in top_indices_ordonnes]
    return " ".join(reponse_phrases)


# ══════════════════════════════════════════════════════════════════
# MODE LLM (optionnel)
# ══════════════════════════════════════════════════════════════════

PROMPT_SYSTEME = """Tu es un assistant expert en recherche documentaire.
Tu réponds UNIQUEMENT en t'appuyant sur les extraits de documents fournis.
Si la réponse ne se trouve pas dans les extraits, dis-le explicitement.
Sois précis, complet et concis. Réponds en français."""


def construire_prompt(requete: str, chunks_recuperes: List[Dict]) -> str:
    """Construit le prompt pour LLM avec contexte numéroté."""
    contexte_parts = []
    for c in chunks_recuperes:
        header = (
            f"[Source {c['rank']}: {c['doc_name']}, "
            f"page {c['page']}, section: {c.get('section','N/A')}]"
        )
        contexte_parts.append(f"{header}\n{c.get('texte_brut', '')}")

    contexte = "\n\n---\n\n".join(contexte_parts)

    return f"""{PROMPT_SYSTEME}

EXTRAITS DE DOCUMENTS :
{contexte}

QUESTION : {requete}

RÉPONSE :"""


def generer_reponse_llm_inference_ai(requete: str,
                                     chunks_recuperes: List[Dict],
                                     timeout: int = 120) -> str:
    """
    Génère réponse via inférence.ai (gratuit, MIT, open source).
    
    Alternative gratuite à Ollama: pas de GPU local requis.
    """
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        print("[LLM] urllib non disponible, fallback extraction")
        return generer_reponse_extraction(requete, chunks_recuperes)

    prompt = construire_prompt(requete, chunks_recuperes)

    # Exemple avec inférence.ai API
    url = "https://api.infer.a-i.ai/v1/completions"
    
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.3,
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
            if "choices" in data:
                return data["choices"][0].get("text", "").strip()
            return generer_reponse_extraction(requete, chunks_recuperes)
    except Exception as e:
        print(f"[LLM] {e}. Fallback extraction.")
        return generer_reponse_extraction(requete, chunks_recuperes)


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATEUR UNIFIÉ
# ══════════════════════════════════════════════════════════════════

def generer_reponse(requete: str,
                   chunks_recuperes: List[Dict],
                   mode: str = "extraction",  # "extraction" ou "llm-inference"
                   **kwargs) -> str:
    """
    Point d'entrée unifié pour génération.
    
    Mode "extraction": ultra-rapide, sans GPU, sans dépendance externe
    Mode "llm-inference": via LLM public, gratuit
    """
    if mode.lower() == "llm-inference":
        return generer_reponse_llm_inference_ai(requete, chunks_recuperes, **kwargs)
    else:
        return generer_reponse_extraction(requete, chunks_recuperes, **kwargs)


# ══════════════════════════════════════════════════════════════════
# FORMATAGE JSON CHALLENGE EVALLLM 2026
# ══════════════════════════════════════════════════════════════════

def formater_resultat_question(qid: str,
                               question: str,
                               chunks_recuperes: List[Dict],
                               reponse: str,
                               parametres: Optional[Dict] = None) -> Dict:
    """Formate résultat selon schéma EvalLLM 2026."""
    
    retrieved = []
    for chunk in chunks_recuperes:
        retrieved.append({
            "rank": chunk["rank"],
            "doc_name": chunk["doc_name"],
            "page": chunk["page"],
            "metadata": {
                "section": chunk.get("section", ""),
                "score_rrf": chunk.get("score_rrf", 0.0),
                "score_reranker": chunk.get("score_reranker", 0.0),
                "nb_pages_doc": chunk.get("nb_pages_doc", 0),
                "regime": chunk.get("regime", ""),
            }
        })

    return {
        "qid": qid,
        "question": question,
        "retrieved": retrieved,
        "answer": reponse,
        "metadata": {"parametres": parametres or {}},
    }


def generer_soumission_json(
    resultats_questions: List[Dict],
    run_id: str = "RAG-Challenge-EvalLLM2026-HF-Online",
    parametres_globaux: Optional[Dict] = None,
) -> Dict:
    """Génère structure JSON complète de soumission."""
    
    return {
        "run_id": run_id,
        "parameters": parametres_globaux or {
            "modele_embedding": "sentence-transformers/all-MiniLM-L6-v2",
            "modele_reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k_initial": 20,
            "top_k_final": 10,
            "rrf_k": 60,
            "mode_generation": "extraction (zéro GPU, zéro LLM)",
        },
        "results": resultats_questions,
    }


def sauvegarder_soumission(soumission: Dict, chemin: Path) -> None:
    """Sauvegarde soumission JSON."""
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(soumission, f, ensure_ascii=False, indent=2)
    print(f"✓ Soumission: {chemin}")
    print(f"  └─ {len(soumission['results'])} questions")


if __name__ == "__main__":
    print("generator_hf.py: 100% Hugging Face")
