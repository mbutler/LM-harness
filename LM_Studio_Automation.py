from lm_harness import LocalLMHarness

# Initialize once. All batch methods resume automatically if interrupted —
# just rerun the script and it picks up where it left off.
harness = LocalLMHarness(
    timeout=600,            # abort a hung generation instead of blocking forever
    max_retries=3,          # transient failures retry with exponential backoff
    max_output_tokens=1024, # cap generation length to protect the KV cache
    max_input_chars=20000,  # truncate oversized inputs to avoid context overflow
    reasoning="off",        # skip the reasoning scratchpad for strict extraction jobs
    workers=4,              # concurrent requests — match your server's "parallel" setting
)

# Define your static, heavily-cached instructions
SYSTEM_PROMPT = """
You are a strict data extraction tool.
Extract the names of any companies mentioned in the input text.
Return ONLY a comma-separated list. Do not use conversational filler.
"""

# Example 1: Crunch a CSV
harness.process_csv(
    input_file="raw_feedback.csv", 
    output_file="extracted_feedback.csv", 
    target_col="customer_comment", 
    system_prompt=SYSTEM_PROMPT
)

# Example 2: Crunch a folder of documents
harness.process_directory(
    input_dir="./raw_transcripts",
    output_dir="./processed_summaries",
    system_prompt="Summarize this transcript in 3 bullet points."
)

# Example 3: Stream a large JSONL dataset (constant memory, per-line resume)
# harness.process_jsonl(
#     input_file="big_dataset.jsonl",
#     output_file="big_dataset_out.jsonl",
#     target_key="text",
#     system_prompt=SYSTEM_PROMPT,
# )