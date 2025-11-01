import json
import re
from annotators.base import AnnotatorStrategy
from config.prompts import BATCH_PROMPT_TEMPLATE
import fitz  # PyMuPDF
from typing import List, Dict

# <-- NEW: Import ONNX and the HF tokenizer for chat templating
import onnxruntime_genai as og
from transformers import AutoTokenizer
# --------------------------------------------------------------
#  ONNX Phi-3 strategy – Processes a full batch in one LLM call
# --------------------------------------------------------------
class Phi3ONNXStrategy(AnnotatorStrategy):
    """
    Uses the ONNX Phi-3-mini-4k-instruct model via onnxruntime-genai.
    This version processes an entire batch of sentences in a single prompt
    to provide context to the model, improving label accuracy.
    """

    def __init__(self, onnx_folder: str):
        print(f"[INFO] Loading Phi-3 ONNX model from {onnx_folder} ...")
        self.model = og.Model(onnx_folder)
        self.tokenizer = og.Tokenizer(self.model)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )
        print("[INFO] ONNX model loaded.")

    def _build_batch_prompt(self, texts: List[str]) -> str:
        """Formats a list of sentences into a single, numbered prompt."""
        formatted_sentences = "\n".join(
            f"{i+1}. \"{s}\"" for i, s in enumerate(texts)
        )
        return BATCH_PROMPT_TEMPLATE.format(
            num_sentences=len(texts),
            formatted_sentences=formatted_sentences
        )

    def _generate_fallback_response(self, num_texts: int, error_msg: str) -> List[Dict]:
        """Creates a default 'none' response for a batch when parsing fails."""
        print(f"[ERROR] LLM response parsing failed: {error_msg}")
        return [
            {"category": "none", "justification": f"Error: {error_msg}"}
            for _ in range(num_texts)
        ]

    def classify_batch(self, chunks: List[str]) -> List[Dict]:
        """
        Takes a list of cleaned sentences, sends them to the LLM in a single
        contextual prompt, and returns a list of classification dictionaries.
        """
        if not chunks:
            return []

        prompt = self._build_batch_prompt(chunks)

        # ---- HF chat template (keeps the same format the model expects)
        chat = [{"role": "user", "content": prompt}]
        input_text = self.hf_tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )

        # ---- Encode with the ONNX tokenizer
        input_ids = self.tokenizer.encode(input_text)
        input_len = len(input_ids)

        # ---- CRITICAL: Calculate max_length for a batch response.
        # We need space for N JSON objects. Let's be generous.
        # Estimate ~80 tokens per JSON object + a 256 token buffer.
        expected_output_len = (len(chunks) * 80) + 256
        max_total_length = min(input_len + expected_output_len, 4096)

        if max_total_length <= input_len:
             return self._generate_fallback_response(
                len(chunks), f"Input too long ({input_len} tokens) for model."
             )

        # ---- Generation parameters
        params = og.GeneratorParams(self.model)
        params.set_search_options(
            max_length=max_total_length,
            temperature=0.0,
            do_sample=False,
        )
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_ids)

        # ---- Generate token-by-token until EOS
        generated_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            generated_tokens.append(token)

        # ---- Decode
        answer = self.tokenizer.decode(generated_tokens).strip()

        # ---- Robust JSON List extraction and validation
        try:
            # Find the JSON array within the model's output
            json_str_match = re.search(r'\[\s*\{.*\}\s*\]', answer, re.S)
            if not json_str_match:
                raise ValueError("No JSON array found in the response.")

            json_str = json_str_match.group(0)
            data = json.loads(json_str)

            # Validate the response
            if not isinstance(data, list):
                raise TypeError("Response is not a list.")
            if len(data) != len(chunks):
                raise ValueError(f"Expected {len(chunks)} items, but got {len(data)}.")

            # Check if each item is a valid dictionary
            for item in data:
                if not isinstance(item, dict) or not {"category", "justification"} <= item.keys():
                    raise ValueError("List item is not a valid dictionary with required keys.")

            return data

        except Exception as e:
            # If anything goes wrong, return a fallback response for the whole batch
            return self._generate_fallback_response(
                len(chunks), f"Parse error: {e}. Raw output: {answer[:200]}..."
            )
