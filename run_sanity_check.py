"""
Sanity check: runs all four processors against the synthetic data in
test_data/. Requires LM Studio to be running with the model loaded.

Outputs land in test_data/output/. Rerunning is safe — resume kicks in
and completed work is skipped, so to test the resume behavior itself,
just Ctrl+C mid-run and launch it again.
"""
import os
from lm_harness import LocalLMHarness, one_of

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

# reasoning="off": these are strict extraction/classification/summary jobs —
# the reasoning scratchpad just burns tokens (it counts against
# max_output_tokens and can starve out the actual answer). Set to None to
# use the model default, "on" to force it.
harness = LocalLMHarness(
    timeout=600,
    max_retries=3,
    max_output_tokens=512,
    max_input_chars=20000,
    reasoning="off",
    workers=4,        # match the "parallel" slot count of the loaded model
    chunk_chars=3000, # small on purpose so call_004_long.txt exercises map-reduce
)

EXTRACT_PROMPT = """
You are a strict data extraction tool.
Extract the names of any companies mentioned in the input text.
Return ONLY a comma-separated list. If none are mentioned, return NONE.
Do not use conversational filler.
"""

print("=" * 60)
print("1/5  CSV: multi-prompt — companies + validated sentiment per row")
print("=" * 60)
harness.process_csv(
    input_file=os.path.join(BASE, "raw_feedback.csv"),
    output_file=os.path.join(OUT, "extracted_feedback.csv"),
    target_col="customer_comment",
    # Multi-prompt mode: each row gets one output column per prompt.
    # A dict value can be a plain prompt or a (prompt, validator) tuple.
    system_prompt={
        "companies": EXTRACT_PROMPT,
        "sentiment": (
            "Classify the sentiment of this feedback as exactly one word: POSITIVE, NEGATIVE, or MIXED.",
            one_of("POSITIVE", "NEGATIVE", "MIXED"),
        ),
    },
)

print("=" * 60)
print("2/5  JSON list: company extraction from support tickets")
print("=" * 60)
harness.process_json_list(
    input_file=os.path.join(BASE, "raw_feedback.json"),
    output_file=os.path.join(OUT, "extracted_tickets.json"),
    target_key="customer_comment",
    system_prompt=EXTRACT_PROMPT,
)

print("=" * 60)
print("3/5  JSONL: sentiment classification of reviews")
print("=" * 60)
harness.process_jsonl(
    input_file=os.path.join(BASE, "raw_feedback.jsonl"),
    output_file=os.path.join(OUT, "classified_reviews.jsonl"),
    target_key="text",
    system_prompt="Classify the sentiment of this review as exactly one word: POSITIVE, NEGATIVE, or MIXED.",
    # Reject any output that isn't exactly one of the labels; the model is
    # shown its bad answer and asked to correct it (validate_retries times).
    validator=one_of("POSITIVE", "NEGATIVE", "MIXED"),
)

print("=" * 60)
print("4/5  Directory: transcript summarization")
print("=" * 60)
harness.process_directory(
    input_dir=os.path.join(BASE, "raw_transcripts"),
    output_dir=os.path.join(OUT, "processed_summaries"),
    system_prompt="Summarize this support call transcript in 3 bullet points.",
)

print("=" * 60)
print("5/5  Images: color identification (vision)")
print("=" * 60)
harness.process_image_directory(
    input_dir=os.path.join(BASE, "images"),
    output_dir=os.path.join(OUT, "image_labels"),
    system_prompt="Identify the dominant solid color of the image. Answer with exactly one word: RED, GREEN, or BLUE.",
    validator=one_of("RED", "GREEN", "BLUE"),
)

print()
print(f"All done. Inspect results in: {OUT}")
