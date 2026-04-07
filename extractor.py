"""
extractor.py
============
Extraction PDF page par page avec PyMuPDF.
Enrichissement uniforme : [DOC] [SECTION] [PAGE] pour tous les documents,
quelle que soit leur taille (1 à 60+ pages).

Stratégie de chunking adaptatif selon nb_pages :
  - 1–3   pages  : chunk = document entier  (1 entrée d'index)
  - 4–15  pages  : chunk = page physique    (1 entrée par page)
  - > 15  pages  : chunk = page + overlap   (200 tokens de la page précédente
                                             + 200 tokens de la page suivante)
Invariant challenge : chaque entrée conserve (doc_name, page_physique).
"""

import fitz  # PyMuPDF
import re
import statistics
from pathlib import Path
from typing import List, Dict, Optional


# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
SEUIL_TITRE_RATIO   = 1.15   # taille > médiane * ratio → probable titre/section
LONGUEUR_MAX_TITRE  = 100    # caractères max pour un titre de section
LONGUEUR_MIN_TITRE  = 3      # caractères min pour être considéré titre
TOKENS_OVERLAP      = 200    # nb tokens approximatif pour l'overlap inter-pages
SEUIL_PAGE_VIDE     = 80     # nb de caractères min pour qu'une page soit indexée


# ──────────────────────────────────────────────
# Utilitaires texte
# ──────────────────────────────────────────────
def approx_tokens(texte: str) -> int:
    """Approximation rapide : 1 token ≈ 4 caractères (sans tokenizer)."""
    return max(1, len(texte) // 4)


def tronquer_tokens(texte: str, nb_tokens: int, depuis_fin: bool = False) -> str:
    """
    Retourne ~nb_tokens tokens depuis le début (ou la fin si depuis_fin=True).
    Conserve les mots entiers.
    """
    nb_chars = nb_tokens * 4
    if depuis_fin:
        fragment = texte[-nb_chars:]
        # Aligner sur un espace pour ne pas couper un mot
        idx = fragment.find(" ")
        return fragment[idx + 1:] if idx != -1 else fragment
    else:
        fragment = texte[:nb_chars]
        idx = fragment.rfind(" ")
        return fragment[:idx] if idx != -1 else fragment


def nettoyer_texte(texte: str) -> str:
    """Supprime les artefacts d'extraction PDF courants."""
    texte = re.sub(r'\n{3,}', '\n\n', texte)          # sauts de ligne excessifs
    texte = re.sub(r'[ \t]{2,}', ' ', texte)           # espaces multiples
    texte = re.sub(r'-\n(\w)', r'\1', texte)            # mots coupés en fin de ligne
    return texte.strip()


# ──────────────────────────────────────────────
# Détection de la taille médiane du corps de texte
# ──────────────────────────────────────────────
def taille_corps_mediane(doc: fitz.Document) -> float:
    """
    Calcule la taille de police médiane du corpus documentaire.
    Utilisée comme référence pour détecter les titres de section.
    Heuristique : les titres ont une taille > médiane * SEUIL_TITRE_RATIO.
    """
    tailles = []
    for page in doc:
        for bloc in page.get_text("dict")["blocks"]:
            if bloc.get("type") != 0:
                continue
            for ligne in bloc["lines"]:
                for span in ligne["spans"]:
                    if span["text"].strip():
                        tailles.append(round(span["size"], 1))
    if not tailles:
        return 10.0
    return statistics.median(tailles)


# ──────────────────────────────────────────────
# Extraction de la section courante
# ──────────────────────────────────────────────
def extraire_sections_par_page(doc: fitz.Document) -> Dict[int, str]:
    """
    Parcourt le document et construit un mapping {page_idx → section_courante}.
    La section courante d'une page = dernier titre détecté avant ou sur cette page.

    Heuristique de détection de titre :
      1. Taille de police >= médiane * SEUIL_TITRE_RATIO, OU
      2. Texte en gras (flag 16 dans PyMuPDF) + ligne courte, OU
      3. Ligne entièrement en majuscules et courte.

    Retourne un dict {0-based page_idx : "Titre de section"}.
    """
    taille_ref = taille_corps_mediane(doc)
    seuil = taille_ref * SEUIL_TITRE_RATIO

    section_courante = ""
    mapping = {}

    for page_idx, page in enumerate(doc):
        for bloc in page.get_text("dict")["blocks"]:
            if bloc.get("type") != 0:
                continue
            for ligne in bloc["lines"]:
                for span in ligne["spans"]:
                    texte = span["text"].strip()
                    if not texte or len(texte) < LONGUEUR_MIN_TITRE:
                        continue
                    if len(texte) > LONGUEUR_MAX_TITRE:
                        continue

                    taille = span["size"]
                    est_gras = bool(span.get("flags", 0) & 16)
                    est_majuscules = texte.isupper() and len(texte) > 4

                    if taille >= seuil or est_gras or est_majuscules:
                        section_courante = texte

        mapping[page_idx] = section_courante

    return mapping


# ──────────────────────────────────────────────
# Construction du préfixe enrichi
# ──────────────────────────────────────────────
def construire_prefixe(
    doc_name: str,
    section: str,
    page_num: int,     # 1-based (numéro physique)
    total_pages: int,
) -> str:
    """
    Génère le préfixe contextuel uniforme pour tous les documents.
    Format : [DOC: fichier.pdf] [SECTION: titre] [PAGE: n/total]
    Si section vide → [SECTION: ] (neutre pour l'embedding).
    """
    return (
        f"[DOC: {doc_name}] "
        f"[SECTION: {section}] "
        f"[PAGE: {page_num}/{total_pages}]\n"
    )


# ──────────────────────────────────────────────
# Extraction complète d'un PDF → liste de chunks
# ──────────────────────────────────────────────
def extraire_chunks(pdf_path: Path) -> List[Dict]:
    """
    Extrait et enrichit tous les chunks d'un PDF.

    Retourne une liste de dicts :
    {
        "doc_name"      : str,   # nom du fichier (pour JSON challenge)
        "page"          : int,   # numéro physique 1-based (pour JSON challenge)
        "section"       : str,   # section détectée (pour métadonnées)
        "nb_pages_doc"  : int,   # taille totale du document
        "texte_enrichi" : str,   # préfixe + texte → utilisé pour embedding & BM25
        "texte_brut"    : str,   # texte seul → passé au générateur LLM
        "est_vide"      : bool,  # True si page trop courte pour être indexée
    }
    """
    doc_name   = pdf_path.name
    doc        = fitz.open(str(pdf_path))
    nb_pages   = len(doc)
    regime     = "page"

    # Extraction texte brut par page (1-based physique)
    textes_pages: Dict[int, str] = {}
    for page_idx, page in enumerate(doc):
        texte = nettoyer_texte(page.get_text("text"))
        textes_pages[page_idx + 1] = texte  # clé = numéro physique

    # Mapping section par page
    sections = extraire_sections_par_page(doc)  # {0-based: section}

    chunks = []
    for page_num in sorted(textes_pages.keys()):
        page_idx   = page_num - 1
        texte_page = textes_pages[page_num]
        est_vide   = len(texte_page) < SEUIL_PAGE_VIDE
        section    = sections.get(page_idx, "")

        prefixe       = construire_prefixe(doc_name, section, page_num, nb_pages)
        texte_enrichi = prefixe + texte_page

        chunks.append({
            "doc_name"      : doc_name,
            "page"          : page_num,
            "section"       : section,
            "nb_pages_doc"  : nb_pages,
            "regime"        : regime,
            "texte_enrichi" : texte_enrichi,
            "texte_brut"    : texte_page,
            "est_vide"      : est_vide,
        })

    doc.close()
    return chunks


# ──────────────────────────────────────────────
# Extraction de tout le corpus
# ──────────────────────────────────────────────
def extraire_corpus(dossier_corpus: Path) -> List[Dict]:
    """
    Parcourt tous les PDFs d'un dossier et retourne la liste complète des chunks.
    Affiche des statistiques par régime.
    """
    pdfs = sorted(dossier_corpus.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {dossier_corpus}")

    tous_chunks = []
    stats = {"court": 0, "moyen": 0, "long": 0}

    for pdf_path in pdfs:
        try:
            chunks = extraire_chunks(pdf_path)
            tous_chunks.extend(chunks)
            regime = chunks[0]["regime"] if chunks else "?"
            stats[regime] = stats.get(regime, 0) + 1
            n_valides = sum(1 for c in chunks if not c["est_vide"])
            print(f"  [{regime:5s}] {pdf_path.name:40s} "
                  f"→ {len(chunks)} pages, {n_valides} indexables")
        except Exception as e:
            print(f"  [ERREUR] {pdf_path.name}: {e}")

    print(f"\nCorpus: {len(pdfs)} docs | "
          f"court:{stats['court']} moyen:{stats['moyen']} long:{stats['long']} | "
          f"total chunks: {len(tous_chunks)}")
    return tous_chunks


if __name__ == "__main__":
    # Test rapide avec un PDF fictif
    import tempfile, os
    print("extractor.py chargé correctement.")
    print("Usage : from extractor import extraire_corpus")
