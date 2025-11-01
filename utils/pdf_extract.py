import re
import fitz  # PyMuPDF
import nltk
from typing import List, Tuple
from pathlib import Path

# <-- NEW: Import ONNX and the HF tokenizer for chat templating
# ==========================================================
# Text Extraction and Cleaning (Unchanged)
# ==========================================================
def clean_text(text: str) -> str:
    text = re.sub(r'\s*\([^)]*,\s*\d{4}\)', '', text)
    text = re.sub(r'\s*\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_sentence(s: str) -> bool:
    if len(s) < 30: return False
    if sum(c.isalpha() for c in s) / len(s) < 0.7: return False
    if s.lower().strip().startswith(('figure', 'table', 'fig.')): return False
    return True

def extract_and_clean_text_by_sentence(pdf_path: Path) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        sentences = nltk.sent_tokenize(text)
        for s in sentences:
            cleaned_s = clean_text(s)
            if is_valid_sentence(cleaned_s):
                chunks.append((i, s))
    doc.close()
    return chunks