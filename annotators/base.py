from abc import ABC, abstractmethod
from typing import List, Dict
# ==========================================================
# STRATEGY PATTERN FOR MODELS
# ==========================================================
class AnnotatorStrategy(ABC):
    @abstractmethod
    def classify_batch(self, chunks: List[str]) -> List[Dict]:
        pass