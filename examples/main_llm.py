

import argparse
from pathlib import Path

import tqdm

from annotators.llm_annotator import Phi3ONNXStrategy
from utils.highlighting import add_highlights
from utils.pdf_extract import extract_and_clean_text_by_sentence


def main():
    parser = argparse.ArgumentParser(
        description="Highlight research paper sections using Phi-3 ONNX (batch-contextual LLM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the input PDF file."
    )
    parser.add_argument(
        "onnx_folder",
        type=Path,
        help="Path to the ONNX model directory (e.g., .../cuda-int4-rtn-block-32/cuda/cuda-int4-rtn-block-32)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        dest="output_path",
        help="Output PDF path. Default: <input>_annotated_llm.pdf",
        default=None
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=4,
        help="Number of sentences per LLM batch. Larger = more context, slower."
    )

    args = parser.parse_args()

    # === Resolve and validate paths ===
    pdf_path = args.pdf_path.expanduser().resolve()
    onnx_folder = args.onnx_folder.expanduser().resolve()

    if not pdf_path.exists():
        parser.error(f"PDF not found: {pdf_path}")
    if not pdf_path.suffix.lower() == ".pdf":
        parser.error(f"Input must be a .pdf file: {pdf_path}")

    if not onnx_folder.exists():
        parser.error(f"ONNX model folder not found: {onnx_folder}")

    # Default output: input file + "_annotated_llm.pdf"
    output_path = args.output_path or pdf_path.with_name(pdf_path.stem + "_annotated_llm.pdf")
    output_path = output_path.expanduser().resolve()

    # === Pipeline ===
    llm_strategy = Phi3ONNXStrategy(onnx_folder)

    print(f"[INFO] Loading PDF: {pdf_path}")
    chunks = extract_and_clean_text_by_sentence(pdf_path)
    print(f"[INFO] Found {len(chunks)} valid sentences.")

    if not chunks:
        print("[WARN] No valid sentences found. Exiting.")
        return

    annotations = []
    batch_size = max(1, args.batch_size)  # Ensure at least 1

    print(f"[INFO] Classifying in batches of {batch_size}...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="LLM Batches"):
        batch = chunks[i:i + batch_size]
        batch_pages, batch_texts = zip(*batch)

        try:
            batch_results = llm_strategy.classify_batch(list(batch_texts))
        except Exception as e:
            print(f"[ERROR] Batch failed: {e}. Using 'none' fallback.")
            batch_results = [
                {"category": "none", "justification": f"Error: {e}"}
                for _ in batch_texts
            ]

        if len(batch_results) != len(batch_texts):
            print(f"[WARN] Result mismatch: {len(batch_results)} vs {len(batch_texts)}. Padding with 'none'.")
            fallback = {"category": "none", "justification": "Result count mismatch."}
            batch_results = (batch_results + [fallback] * len(batch_texts))[:len(batch_texts)]

        for page_idx, text, result in zip(batch_pages, batch_texts, batch_results):
            annotations.append((page_idx, text, result))

    print(f"[INFO] Saving annotated PDF → {output_path}")
    add_highlights(pdf_path, annotations, output_path)
    print(f"[DONE] Saved: {output_path}")