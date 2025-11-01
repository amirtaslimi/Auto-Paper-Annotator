BATCH_PROMPT_TEMPLATE = """You are a scientist and research paper annotator.
Your task is to label each sentence in the provided numbered list based on its content and contextual meaning within the batch.

Possible categories (aligned with highlight colors):
- "innovation" novel ideas, key claims, originality, improvements
- "related_work" prior work, comparisons, literature discussion
- "reason" explanations or conceptual reasoning
- "method" approaches, architectures, algorithms, or procedural descriptions
- "dataset" data details, collection process, benchmarks used
- "results" performance analysis, metrics, evaluation
- "limitation" shortcomings, weaknesses, challenges
- "future_work" suggestions, open directions, next steps
- "none" → irrelevant or general statements

Analyze the following {num_sentences} sentences:
{formatted_sentences}

Return a **valid JSON list** with one entry per sentence.
Each entry must include:
  - "category": one of the exact labels above only.
  - "justification": a short explanation (1–2 sentences) for why that category was chosen

Example format:
[
  {{"category": "methodology", "justification": "Describes the architecture and optimization process."}},
  {{"category": "related_work", "justification": "Mentions previous approaches and comparison."}}
]
"""