from annotators.base import AnnotatorStrategy
from config.labels import LABELS
import fitz  # PyMuPDF
from typing import List, Dict
from tqdm import tqdm
from transformers import pipeline

class ZeroShotSentence(AnnotatorStrategy):
    """Sentence-level zero-shot classification that obeys the batch interface."""
    
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        print(f"[INFO] Loading Zero-Shot classification model: {model_name} ...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0,                     # GPU if available
            batch_size=1,                 # we will batch manually
        )
        self.hypothesis_template = "This sentence is about {}."
        print("[INFO] Model loaded.")

    # ------------------------------------------------------------------ #
    # PRIVATE – single-sentence classification (unchanged from your code)
    # ------------------------------------------------------------------ #
    def _classify_sentence(self, text: str) -> Dict:
        result = self.classifier(
            text,
            LABELS,
            hypothesis_template=self.hypothesis_template,
            multi_label=False,
        )
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        category_key = top_label.replace(" ", "_")

        return {
            "category": category_key,
            "confidence": top_score,
            "evidence": [text],
            "justification": f"Zero-shot classified as '{top_label}' with {top_score:.2f} confidence."
        }

    # ------------------------------------------------------------------ #
    # PUBLIC – batch interface required by AnnotatorStrategy
    # ------------------------------------------------------------------ #
    def classify_batch(self, chunks: List[str]) -> List[Dict]:
        """
        `chunks` → list of *individual* sentences (already cleaned & filtered).
        Returns a list of dicts, one per sentence, in the same order.
        """
        annotations = []
        # tqdm gives you a nice progress bar for long papers
        for sentence in tqdm(chunks, desc="Zero-shot NLI", leave=False):
            ann = self._classify_sentence(sentence)
            annotations.append(ann)
        return annotations