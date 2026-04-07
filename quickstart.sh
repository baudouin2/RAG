#!/bin/bash
#
# quickstart.sh
# =============
# Démarrage rapide du pipeline RAG pour le corpus réel 2026-04-07
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS_DIR="/home/baudouin/Downloads/Download 2026-04-07T00-49-49-273Z/Corpus_raw"
QUESTIONS_FILE="/home/baudouin/Downloads/Download 2026-04-07T00-49-49-273Z/sample_queries.json"
INDEX_DIR="${SCRIPT_DIR}/index_real"
OUTPUT_FILE="${SCRIPT_DIR}/soumission_$(date +%Y-%m-%d_%Hh%M).json"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   RAG Pipeline — Quickstart avec corpus réel 2026-04-07       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Vérifier les prérequis
echo "[✓] Vérification des fichiers..."

if [ ! -d "$CORPUS_DIR" ]; then
    echo "❌ Corpus introuvable : $CORPUS_DIR"
    exit 1
fi
echo "  ✓ Corpus trouvé : $(ls "$CORPUS_DIR"/*.pdf 2>/dev/null | wc -l) PDFs"

if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "❌ Fichier de questions introuvable : $QUESTIONS_FILE"
    exit 1
fi
echo "  ✓ Questions trouvées : $(jq '.results | length' "$QUESTIONS_FILE" 2>/dev/null || echo "?") requêtes"

# Étape 1 : Indexation
if [ ! -d "$INDEX_DIR" ]; then
    echo ""
    echo "[1/2] 📊 INDEXATION DU CORPUS"
    echo "      Création de l'index FAISS + BM25..."
    echo ""
    
    python3 "$SCRIPT_DIR/pipeline_hf.py" \
        --mode index \
        --corpus "$CORPUS_DIR" \
        --index "$INDEX_DIR" \
        --no-rerank
    
    echo ""
    echo "      ✓ Index créé dans : $INDEX_DIR"
else
    echo ""
    echo "[1/2] 📊 Index existant trouvé"
    echo "      Réutilisation : $INDEX_DIR"
fi

# Afficher les stats de l'index
echo ""
echo "[Info] Statistiques de l'index :"
python3 "$SCRIPT_DIR/inspect_vectors.py" 2>/dev/null | head -40

# Étape 2 : Réponses
echo ""
echo "[2/2] 🔍 REQUÊTES ET GÉNÉRATION"
echo "      Traitement des questions..."
echo ""

python3 "$SCRIPT_DIR/pipeline_hf.py" \
    --mode run \
    --questions "$QUESTIONS_FILE" \
    --index "$INDEX_DIR" \
    --output "$OUTPUT_FILE" \
    --top_k 10 \
    --generation extraction

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Pipeline terminé avec succès                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Résultats :"
echo "   Index   : $INDEX_DIR"
echo "   Sortie  : $OUTPUT_FILE"
echo ""
echo "📊 Stats JSON :" && echo "✓ Index généré et prêt"
echo ""
echo "🔍 Inspecter le résultat :"
echo "   cat $OUTPUT_FILE | head -100"
echo ""
