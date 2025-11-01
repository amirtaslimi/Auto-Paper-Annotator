import argparse
from pathlib import Path

import tqdm

from annotators.nli_annotator import ZeroShotSentence
from utils.highlighting import add_highlights
from utils.pdf_extract import extract_and_clean_text_by_sentence

def main():
    parser = argparse.ArgumentParser(
        description="Highlight research paper sections using Zero-Shot NLI (sentence-level).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the input PDF file."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        dest="output_path",
        help="Output PDF path. Default: <input>_sentence_annotated.pdf",
        default=None
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Number of sentences per zero-shot batch. Larger = faster, less memory per call."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        help="Hugging Face zero-shot classification model."
    )

    args = parser.parse_args()

    # === Resolve and validate paths ===
    pdf_path = args.pdf_path.expanduser().resolve()

    if not pdf_path.exists():
        parser.error(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        parser.error(f"Input must be a .pdf file: {pdf_path}")

    # Default output
    output_path = args.output_path or pdf_path.with_name(pdf_path.stem + "_sentence_annotated.pdf")
    output_path = output_path.expanduser().resolve()

    # === Pipeline ===
    print(f"[INFO] Loading PDF: {pdf_path}")
    page_sentence_tuples = extract_and_clean_text_by_sentence(pdf_path)
    sentences = [sent for _page_idx, sent in page_sentence_tuples]

    if not sentences:
        print("[WARN] No valid sentences found in the PDF.")
        return

    print(f"[INFO] Found {len(sentences)} valid sentences. Starting zero-shot classification...")

    # Initialize strategy with optional model override
    strategy = ZeroShotSentence(model_name=args.model)

    batch_size = max(1, args.batch_size)
    annotations = []

    print(f"[INFO] Classifying in batches of {batch_size}...")
    for i in tqdm(range(0, len(sentences), batch_size), desc="NLI Batches"):
        batch_sentences = sentences[i:i + batch_size]
        batch_results = strategy.classify_batch(batch_sentences)

        # Safety: align lengths
        if len(batch_results) != len(batch_sentences):
            print(f"[WARN] Batch size mismatch: {len(batch_results)} vs {len(batch_sentences)}. Padding.")
            fallback = {"category": "none", "confidence": 0.0, "justification": "Batch mismatch."}
            batch_results = (batch_results + [fallback] * len(batch_sentences))[:len(batch_sentences)]

        # Reattach page numbers
        batch_page_tuples = page_sentence_tuples[i:i + batch_size]
        for (page_idx, sent), ann in zip(batch_page_tuples, batch_results):
            annotations.append((page_idx, sent, ann))

    print(f"[INFO] Saving annotated PDF → {output_path}")
    add_highlights(pdf_path, annotations, output_path)
    print(f"[SUCCESS] Annotated PDF saved to: {output_path}")