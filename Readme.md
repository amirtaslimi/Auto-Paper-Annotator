# Semantic Paper Highlighter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An AI-powered tool that automatically reads and color-codes research papers (PDFs) to help you quickly identify contributions, methodologies, experimental results, and other key sections. Supports both powerful local LLMs for high-quality analysis and lightweight NLI models for speed.

---

## ✨ Demo

Imagine opening a dense, 20-page research paper and instantly seeing it transformed into a color-coded summary. That's what this tool does.

*   <span style="color: #6278f4ff">**Innovation & Contributions**</span> are marked in red.
*   <span style="color: #9933CC">**Methods & Methodology**</span> are highlighted in purple.
*   <span style="color: #40CC66">**Results & Experiments**</span> stand out in green.
*   And so on...

![Demo GIF of the Semantic Paper Highlighter in action](images/demo.gif)
*(**Note:** You'll need to create a `demo.gif` and place it in the `images` folder for this to display)*

## 🚀 Features

*   **Semantic Highlighting:** Goes beyond keyword search to understand the *meaning* of each sentence.
*   **Color-Coded Categories:** Assigns a unique color to each category (e.g., `method`, `results`, `limitation`) for easy visual scanning.
*   **Dual-Backend Strategy:**
    *   🧠 **LLM Annotator:** Uses a local, quantized LLM (like Phi-3) via ONNX Runtime for high-accuracy, contextual annotations. **Your data never leaves your machine.**
    *   ⚡ **NLI Annotator:** Employs a fast Zero-Shot Natural Language Inference (NLI) model for quick processing without needing a powerful GPU.
*   **PDF to PDF:** Takes a PDF as input and generates a new, annotated PDF file with hover-text justifications.
*   **Command-Line Interface:** Easy to use and integrate into scripts.

## 🤔 How It Works

The pipeline is straightforward but powerful:

1.  **PDF Parsing:** The input PDF is read, and all text is extracted page by page using PyMuPDF.
2.  **Sentence Segmentation:** The text is cleaned of artifacts (like citation marks) and split into individual sentences using NLTK.
3.  **Semantic Classification:** Each sentence is classified using one of the chosen strategies:
    *   The **LLM Annotator** sends batches of sentences to a local ONNX model, asking it to return a JSON object identifying the category for each sentence based on the surrounding context.
    *   The **NLI Annotator** checks each sentence against a list of candidate labels (e.g., "This sentence is about methodology") and picks the best fit.
4.  **Highlight Generation:** A new PDF is created where each classified sentence is highlighted with its corresponding color. You can hover over a highlight to see the model's justification.

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/semantic-highlighter.git
    cd semantic-highlighter
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK data:**
    The tool uses NLTK for sentence tokenization. Run this command once to download the necessary data:
    ```bash
    python -c "import nltk; nltk.download('punkt')"
    ```

4.  **(Optional but Recommended) Download an ONNX LLM:**
    To use the powerful LLM annotator, you need a compatible model. We recommend Microsoft's **Phi-3-mini-instruct** in ONNX format.
    *   You can find pre-converted models on the [ONNX Runtime GenAI Model Hub](https://huggingface.co/collections/microsoft/phi-3-onnx-models-662651125237c352701b2585).
    *   Download the model files and place them in a dedicated folder.

## 📖 Usage

This is a command-line tool. You can run the annotators from your terminal.

### 1. Using the LLM Annotator (High Quality)

This method provides the best results by understanding the context between sentences. It requires a path to your PDF and the downloaded ONNX model folder.

**Basic Command:**
```bash
python examples/main_llm.py /path/to/my_paper.pdf /path/to/my_onnx_model
```
This will create a new file named `my_paper_annotated_llm.pdf` in the same directory.

**Advanced Command with Options:**
You can specify an output path and change the batch size.
```bash
python examples/main_llm.py "My Research/Attention Is All You Need.pdf" "Models/Phi-3-ONNX/" -o "Annotated/Attention_annotated.pdf" -b 8
```
*   `-o, --output`: Specifies a custom path and filename for the output PDF.
*   `-b, --batch-size`: Sets the number of sentences to process in each LLM call. Larger values give the model more context but require more memory. Default is 4.

To see all available options, run:
```bash
python examples/main_llm.py --help
```

### 2. Using the NLI Annotator (Fast)

This method is faster and doesn't require a separate large model download, but may be less accurate than the LLM. It's great for quick scans.

**Basic Command:**
```bash
python examples/main_nli.py /path/to/my_paper.pdf
```
This will create a new file named `my_paper_sentence_annotated.pdf` in the same directory.

**Advanced Command with Options:**
You can specify an output path and even try a different NLI model from Hugging Face.
```bash
python examples/main_nli.py "My Research/BERT.pdf" -o "Annotated/BERT_annotated.pdf" --model "facebook/bart-large-mnli"
```
*   `-o, --output`: Specifies a custom path for the output PDF.
*   `--model`: Lets you specify a different zero-shot classification model from the Hugging Face Hub.
*   `-b, --batch-size`: Sets the inference batch size for the NLI model. Larger is generally faster on GPU. Default is 32.

To see all available options, run:
```bash
python examples/main_nli.py --help
```

## 🛠️ Customization

While many options are available via the command line, you can still customize the core logic in the `config/` directory.

*   **Labels & Colors:** Edit `config/labels.py` to add, remove, or change the annotation categories and their associated highlight colors.
*   **LLM Prompts:** Modify `config/prompts.py` to change the instructions given to the Large Language Model. You can fine-tune the prompt to better suit your domain or desired output format.

## 📂 Project Structure

```
├── Readme.md               # You are here!
├── annotators/             # Contains the classification strategies (LLM, NLI)
│   ├── base.py             # Defines the abstract base class for annotators
│   ├── llm_annotator.py    # The powerful ONNX LLM strategy
│   └── nli_annotator.py    # The fast Zero-Shot NLI strategy
├── config/                 # All user-configurable files
│   ├── labels.py           # Define categories and colors
│   └── prompts.py          # Define the prompt for the LLM
├── examples/               # Executable CLI scripts to run the pipeline
│   ├── main_llm.py         # Main script for the LLM annotator
│   └── main_nli.py         # Main script for the NLI annotator
├── images/                 # For storing demo GIFs and images
├── requirements.txt        # Project dependencies
└── utils/                  # Helper modules
    ├── highlighting.py     # Logic for adding highlights to PDFs
    └── pdf_extract.py      # Logic for extracting and cleaning text
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/semantic-highlighter/issues).

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

