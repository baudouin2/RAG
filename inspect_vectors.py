#!/usr/bin/env python3
"""
inspect_vectors.py
==================
Inspecte les stats de l'index FAISS et les chunks stockés.
"""

import sys
import json
from pathlib import Path
import faiss
import numpy as np

def inspect_index(index_dir: Path = Path("index")):
    """Affiche les statistiques complètes de l'index."""
    
    if not index_dir.exists():
        print(f"❌ Erreur : Dossier '{index_dir}' non trouvé")
        print("D'abord : python pipeline.py --mode index ...")
        return False
    
    # Vérifier les 4 fichiers obligatoires
    fichiers_requis = ["faiss.index", "bm25.pkl", "chunks.json", "mapping.json"]
    fichiers_manquants = [f for f in fichiers_requis if not (index_dir / f).exists()]
    
    if fichiers_manquants:
        print(f"❌ Fichiers manquants : {fichiers_manquants}")
        return False
    
    print("=" * 70)
    print("📊 INSPECTION DE L'INDEX FAISS + VECTEURS")
    print("=" * 70)
    
    # 1. Charger les métadonnées
    print("\n[1] Chargement des métadonnées...")
    with open(index_dir / "chunks.json") as f:
        chunks = json.load(f)
    print(f"  ✓ {len(chunks)} chunks chargés")
    
    with open(index_dir / "mapping.json") as f:
        mapping = json.load(f)
    print(f"  ✓ {len(mapping)} chunks valides (non-vides)")
    
    # 2. Charger l'index FAISS
    print("\n[2] Chargement de l'index FAISS...")
    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    print(f"  ✓ Index FAISS chargé")
    
    # 3. Statistiques FAISS
    print("\n[3] Statistiques FAISS :")
    print(f"  - Type : IndexFlatIP (brute-force cosine similarity)")
    print(f"  - Dimension vecteurs : {faiss_index.d}")
    print(f"  - Nombre de vecteurs : {faiss_index.ntotal}")
    
    size_mb = (faiss_index.ntotal * faiss_index.d * 4) / (1024 * 1024)
    print(f"  - Taille en mémoire : {size_mb:.1f} MB")
    print(f"  - Par chunk : {(faiss_index.d * 4 / 1024):.1f} KB")
    
    # 4. Statistiques par document
    print("\n[4] Documents indexés :")
    docs = {}
    vides = 0
    for chunk in chunks:
        if chunk["est_vide"]:
            vides += 1
        else:
            doc_name = chunk["doc_name"]
            docs[doc_name] = docs.get(doc_name, 0) + 1
    
    for doc_name in sorted(docs.keys()):
        count = docs[doc_name]
        nb_pages = chunks[[c for c in chunks if c["doc_name"] == doc_name][0]]["nb_pages_doc"]
        print(f"  - {doc_name:50s} : {count:3d}pp indexées / {nb_pages:3d}pp total")
    
    print(f"\n  Total : {len(chunks)} chunks ({len(mapping)} indexés, {vides} vides)")
    
    # 5. Statistiques de pages vides
    if vides > 0:
        pct = 100 * vides / len(chunks)
        print(f"  ⚠️  Pages vides : {vides} ({pct:.1f}%)")
    
    # 6. Statistiques par régime
    print("\n[5] Distribution par régime de chunking :")
    regimes = {}
    for chunk in chunks:
        regime = chunk.get("regime", "?")
        regimes[regime] = regimes.get(regime, 0) + 1
    
    for regime in sorted(regimes.keys()):
        count = regimes[regime]
        pct = 100 * count / len(chunks)
        print(f"  - {regime:10s} : {count:4d} chunks ({pct:5.1f}%)")
    
    # 7. Statistiques de sections
    print("\n[6] Sections détectées (top 10) :")
    sections = {}
    for chunk in chunks:
        sec = chunk.get("section", "[Aucune]")
        sections[sec] = sections.get(sec, 0) + 1
    
    sections_top10 = sorted(sections.items(), key=lambda x: x[1], reverse=True)[:10]
    for section, count in sections_top10:
        section_disp = (section[:60] + "...") if len(section) > 60 else section
        print(f"  - {section_disp:65s} : {count:3d} chunks")
    
    # 8. Taille disque
    print("\n[7] Taille disque :")
    for fname in fichiers_requis:
        fpath = index_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  - {fname:20s} : {size_mb:8.1f} MB")
    
    total_mb = sum((index_dir / f).stat().st_size for f in fichiers_requis) / (1024 * 1024)
    print(f"  - TOTAL               : {total_mb:8.1f} MB")
    
    # 9. Vérification intégrité
    print("\n[8] Vérification intégrité :")
    if faiss_index.ntotal != len(mapping):
        print(f"  ❌ MISMATCH : FAISS a {faiss_index.ntotal} vecteurs mais mapping en a {len(mapping)}")
    else:
        print(f"  ✓ FAISS et mapping cohérents ({len(mapping)} vecteurs)")
    
    if all(i < len(chunks) for i in mapping):
        print(f"  ✓ Tous les indices de mapping sont valides")
    else:
        print(f"  ❌ Certains indices de mapping sont hors limites!")
    
    print("\n" + "=" * 70)
    print("✅ Inspection terminée\n")
    return True


def exporter_statistics(index_dir: Path = Path("index"), output: Path = Path("index_stats.json")):
    """Exporte les statistiques en JSON pour analyse."""
    
    with open(index_dir / "chunks.json") as f:
        chunks = json.load(f)
    
    with open(index_dir / "mapping.json") as f:
        mapping = json.load(f)
    
    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    
    # Construire les stats
    stats = {
        "timestamp": str(Path.cwd()),
        "index_path": str(index_dir),
        "faiss": {
            "dimension": faiss_index.d,
            "ntotal": faiss_index.ntotal,
            "size_mb": (faiss_index.ntotal * faiss_index.d * 4) / (1024 * 1024),
        },
        "chunks": {
            "total": len(chunks),
            "valides": len(mapping),
            "vides": sum(1 for c in chunks if c["est_vide"]),
        },
        "documents": {},
        "regimes": {},
        "sections_top20": [],
    }
    
    # Docs
    for chunk in chunks:
        doc = chunk["doc_name"]
        if doc not in stats["documents"]:
            stats["documents"][doc] = 0
        if not chunk["est_vide"]:
            stats["documents"][doc] += 1
    
    # Regimes
    for chunk in chunks:
        regime = chunk.get("regime", "?")
        if regime not in stats["regimes"]:
            stats["regimes"][regime] = 0
        if not chunk["est_vide"]:
            stats["regimes"][regime] += 1
    
    # Sections top
    sections = {}
    for chunk in chunks:
        sec = chunk.get("section", "[Aucune]")
        sections[sec] = sections.get(sec, 0) + 1
    
    stats["sections_top20"] = [
        {"name": sec, "count": count}
        for sec, count in sorted(sections.items(), key=lambda x: x[1], reverse=True)[:20]
    ]
    
    with open(output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Statistiques exportées : {output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("index_stats.json")
        exporter_statistics(output=output_path)
    else:
        inspect_index()
