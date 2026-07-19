# LocalLMHarness

A lightweight, strictly stateless Python harness for running high-volume batch data processing through a locally hosted LLM via [LM Studio](https://lmstudio.ai)'s v1 REST API.

Built for the realities of a large reasoning model on constrained unified memory: every request is fully stateless (`"store": false`, so the server never accumulates background context), the token-heavy reasoning scratchpad is stripped from every response, and all batch jobs are crash-tolerant — interrupt a 10,000-row run at row 9,999 and rerunning resumes exactly where it stopped.

## Requirements

- Python 3.8+ with `requests` (`pip install requests`)
- LM Studio running with its local server enabled (default `http://localhost:1234`) and a model loaded

## Quick start

```python
from lm_harness import LocalLMHarness, one_of, valid_json

harness = LocalLMHarness(
    model="google/gemma-4-26b-a4b-qat",
    timeout=600,            # abort a hung generation instead of blocking forever
    max_retries=3,          # transient failures retry with exponential backoff
    max_output_tokens=1024, # cap generation length to protect the KV cache
    max_input_chars=20000,  # truncate oversized inputs to avoid context overflow
    reasoning="off",        # skip the reasoning scratchpad for strict extraction jobs
    workers=4,              # concurrent requests — match your server's "parallel" setting
)

harness.process_csv(
    input_file="raw_feedback.csv",
    output_file="extracted_feedback.csv",
    target_col="customer_comment",
    system_prompt="Extract the companies mentioned. Return ONLY a comma-separated list, or NONE.",
)
```

A full working demo lives in `run_sanity_check.py`, which exercises every feature against the synthetic data in `test_data/`. Rerunning it is always safe — completed work is skipped. To watch resume in action, Ctrl+C mid-run and launch it again.

## Processors

All four share the same behavior: incremental durable writes, automatic resume, parallel workers with output kept in input order, and failures isolated per item.

| Method | Input | Output | Best for |
|---|---|---|---|
| `process_csv` | CSV with a target column | Same CSV + output column(s) | Tabular data |
| `process_json_list` | JSON array of objects | Same array + output key(s) | Small/medium JSON datasets (held in memory) |
| `process_jsonl` | JSON Lines, one object per line | JSONL + output key(s) | Large datasets — constant memory, streamed |
| `process_directory` | Folder of text files | One `processed_<name>` file each | Documents, transcripts |
| `process_image_directory` | Folder of images | One `processed_<name>.txt` each | OCR, captioning, classification (vision models) |

```python
harness.process_jsonl("reviews.jsonl", "classified.jsonl", target_key="text",
                      system_prompt="Classify sentiment as exactly one word: POSITIVE, NEGATIVE, or MIXED.",
                      validator=one_of("POSITIVE", "NEGATIVE", "MIXED"))

harness.process_directory("./transcripts", "./summaries",
                          system_prompt="Summarize this call in 3 bullet points.")
```

### Resume (`resume=True`, the default)

Every processor checkpoints as it goes and picks up where it left off when rerun:

- **CSV / JSONL** — each row/line is flushed and fsynced as it completes; on restart, completed records are counted and skipped, and a torn final line from a hard kill is repaired automatically.
- **JSON list** — the output file is atomically checkpointed every `checkpoint_every` items (default 1). On resume, only missing keys — or keys marked `[ERROR]`, unless `retry_errors=False` — are regenerated.
- **Directory** — files whose output already exists are skipped; outputs recording a failed attempt are retried. Files are written via temp-file-then-rename, so a half-written output is never mistaken for done.

Pass `resume=False` to start a job over.

### Error handling

A request that fails after all retries never poisons your data as fake model output. In batch mode the item is recorded as a greppable `[ERROR] <reason>` marker, counted, reported at the end — and retried on the next resume where the format allows it. Calling `generate()` directly raises `GenerationError` instead.

## Multi-prompt mode

Any processor's `system_prompt` also accepts a dict, running every prompt against each input in one pass. Dict values are a prompt string or a `(prompt, validator)` tuple:

```python
harness.process_csv("feedback.csv", "enriched.csv", target_col="comment",
    system_prompt={
        "companies": "Extract company names as a comma-separated list, or NONE.",
        "sentiment": ("Classify as exactly one word: POSITIVE, NEGATIVE, or MIXED.",
                      one_of("POSITIVE", "NEGATIVE", "MIXED")),
    })
```

- CSV: one new column per prompt. JSON/JSONL: one key per prompt.
- Directory jobs write a JSON object per file (single-prompt directory jobs still write raw text).
- JSON-list resume regenerates only the missing/failed keys of each object.

## Validated outputs with corrective retry

Pass a `validator` to any processor or to `generate()`. On an invalid output, the model is shown its rejected answer and the reason, and asked to correct itself — up to `validate_retries` times (default 2) before the item is marked `[ERROR]`.

Built-in validator factories:

```python
one_of("POSITIVE", "NEGATIVE", "MIXED")   # output must be exactly one label (case-insensitive)
valid_json(require_keys=["name", "amount"])  # must parse as JSON; tolerates ``` fences
```

Or any plain function: return `None`/`True` for valid; return `False`, a reason string, or raise `ValueError` for invalid.

```python
harness.process_csv(..., validator=lambda out: None if len(out) < 500 else "response too long")
```

## Long documents: map-reduce

`generate_long()` handles inputs of any length: the text is split at natural boundaries (paragraphs → lines → sentences → words, losslessly) into chunks of at most `chunk_chars`, the prompt is mapped over every chunk in parallel, and the partial results are combined into one final answer — hierarchically, over multiple rounds, if the partials are themselves too large. Short inputs fall through to a single plain call, so it is always safe to use.

`process_directory` uses it automatically (`map_reduce=False` restores plain truncation). Validators apply only to the final combined result. Override the combine step's instructions with `combine_prompt=` if the default ("merge these partial results as if the instruction ran on the whole document") doesn't fit the job.

## Images (vision models)

If the loaded model reports `"vision": true` in `/api/v1/models`, images can be attached to any generation — the v1 API takes them as typed input parts, which the harness builds for you:

```python
# Direct: file paths or existing data: URIs
harness.generate("Read the total amount from this receipt.", "Extract the total.",
                 images=["receipt.jpg"])

# Batch: one output file per image; validators and multi-prompt work as usual
harness.process_image_directory(
    input_dir="./scans", output_dir="./labels",
    system_prompt="Identify the dominant color. One word: RED, GREEN, or BLUE.",
    validator=one_of("RED", "GREEN", "BLUE"),
)
```

Files are base64-encoded automatically (`.png`, `.jpg`/`.jpeg`, `.webp`, `.gif`, `.bmp`). Images count toward the context window, so keep the loaded context in mind for high-resolution inputs.

**Audio is not supported.** Even for models whose cards claim audio understanding, LM Studio's v1 API only accepts `text` and `image` input parts (the server rejects anything else), so there is nothing for the harness to call. If you need audio, transcribe first (e.g. with Whisper) and feed the text through a normal processor.

## Constructor reference

| Option | Default | Notes |
|---|---|---|
| `base_url` | `http://localhost:1234/api/v1/chat` | LM Studio v1 chat endpoint |
| `model` | `google/gemma-4-26b-a4b-qat` | Model key as reported by `/api/v1/models` |
| `timeout` | `600` | Read timeout (s); a hung server aborts instead of blocking forever |
| `connect_timeout` | `10` | TCP connect timeout (s) |
| `max_retries` | `3` | Attempts per request (connection errors, timeouts, 5xx) |
| `retry_backoff` | `5.0` | Base delay (s), doubles per retry |
| `max_output_tokens` | `None` | Server-side generation cap — protects the KV cache |
| `max_input_chars` | `None` | Client-side input truncation guard (~4 chars/token) |
| `reasoning` | `None` | `"on"` / `"off"` / `None` (model default) — see below |
| `workers` | `1` | Concurrent requests; match your server's `parallel` slot count |
| `chunk_chars` | `16000` | Map-reduce chunk budget (~4k tokens); keep below `max_input_chars` |
| `validate_retries` | `2` | Corrective attempts after a validator rejection |

## Memory & throughput notes for local models

- **Statelessness** — every payload sends `"store": false` and the response's `reasoning` block is discarded; only the final `message` payload reaches your files. The context window never silently accumulates history between iterations.
- **Reasoning costs output tokens.** Reasoning tokens count against `max_output_tokens`; a reasoning model can burn the entire budget thinking and never emit a message (the harness detects this and says so in the error). For strict extraction/classification, `reasoning="off"` is dramatically faster and cheaper — in testing, a 255-token response dropped to 7 tokens at ~3x speed. Flip it per job: `harness.reasoning = "on"`.
- **Match `workers` to the server.** Check `GET /api/v1/models` → `"parallel"` for the loaded instance and set `workers` to it, never above — each concurrent slot holds its own KV cache. A global semaphore caps total in-flight requests at `workers` even when parallel loops nest (e.g. a directory job map-reducing several files at once). Measured on a 26B model with 4 slots: a 7-row job went from 23.3s serial to 2.7s.
- **Watch the loaded context, not the max.** A model may advertise a 262k max context but be loaded at 8k. Size `chunk_chars` and `max_input_chars` to the loaded value (`"context_length"` in `/api/v1/models`), leaving headroom for the system prompt and the response.

## Files

| File | Purpose |
|---|---|
| `lm_harness.py` | The harness — single file, stdlib + `requests` only |
| `LM_Studio_Automation.py` | Minimal usage example |
| `run_sanity_check.py` | End-to-end demo of every feature against `test_data/` |
| `test_data/` | Synthetic CSV / JSON / JSONL / transcripts, incl. a long multi-topic transcript that exercises map-reduce |
