from config.labels import CATEGORY_COLORS
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
from pathlib import Path

# <-- NEW: Import ONNX and the HF tokenizer for chat templating

# ==========================================================
# PDF Highlighting (Unchanged)
# ==========================================================
def add_highlights(pdf_path: Path, annotations: List[Tuple[int, str, Dict]], output_path: Path):
    doc = fitz.open(pdf_path)
    for page_idx, sentence, result in annotations:
        cat = result.get("category", "none")
        if cat == "none":
            continue

        color = CATEGORY_COLORS.get(cat)
        if not color:
            print(f"[WARN] No color defined for category: '{cat}'. Using gray.")
            color = (0.8, 0.8, 0.8)

        page = doc[page_idx]
        areas = page.search_for(sentence, quads=True)

        if not areas:
            print(f"[WARN] Could not find sentence on page {page_idx+1}: '{sentence[:50]}...'")
            continue

        for quad in areas:
            hl = page.add_highlight_annot(quad)
            hl.set_colors(stroke=color)
            hl.set_info(title=cat, content=result.get("justification", "No justification provided."))
            hl.update()

    doc.save(output_path, deflate=True, garbage=4)
    doc.close()