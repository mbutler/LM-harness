import base64
import json
import csv
import os
import threading
import time
import requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from urllib.parse import urlsplit


class GenerationError(Exception):
    """Raised when the LM Studio API fails after all retry attempts."""
    pass


class LocalLMHarness:
    """
    A lightweight harness for orchestrating local LLM data pipelines via LM Studio.

    Design goals:
      - Strictly stateless: every call sends "store": False so LM Studio never
        accumulates background context in the KV cache.
      - Structural filtering: the split "output" array is parsed and the
        token-heavy "reasoning" scratchpad is dropped; only the final
        "message" payload survives.
      - Crash-tolerant: all batch processors write incrementally and can
        resume where they left off (resume=True, the default).

    Every processor's `system_prompt` accepts either a single prompt string
    (output lands in 'llm_output') or a dict for multi-prompt mode — each
    input is run through every prompt in one pass:

        prompts = {
            "sentiment": ("Classify as POSITIVE/NEGATIVE/MIXED.", one_of("POSITIVE", "NEGATIVE", "MIXED")),
            "companies": "Extract company names as a comma-separated list.",
        }
        harness.process_csv(..., system_prompt=prompts)  # adds 2 columns

    Dict values are either a prompt string or a (prompt, validator) tuple;
    a `validator=` argument passed alongside applies to keys without their own.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/api/v1/chat",
        model: str = "google/gemma-4-26b-a4b-qat",
        timeout: float = 600.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        retry_backoff: float = 5.0,
        max_output_tokens: Optional[int] = None,
        max_input_chars: Optional[int] = None,
        reasoning: Optional[str] = None,
        workers: int = 1,
        chunk_chars: int = 16000,
        validate_retries: int = 2,
        compat_url: Optional[str] = None,
    ):
        """
        Args:
            timeout: Read timeout in seconds. A hung local server would
                otherwise block the pipeline forever. Generous by default
                because a 26B reasoning model can legitimately think a while.
            connect_timeout: Seconds to wait for the TCP connection.
            max_retries: Attempts per request before raising GenerationError.
                Retries cover connection errors, timeouts, and 5xx responses.
            retry_backoff: Base delay in seconds; doubles each retry.
            max_output_tokens: If set, caps generation length server-side —
                a runaway completion can't chew through the KV cache.
            max_input_chars: If set, inputs are truncated client-side before
                sending. Guards against a single oversized document
                overflowing the context window mid-batch.
            reasoning: "on", "off", or None (model default). For strict
                extraction/classification jobs, "off" is dramatically faster
                and cheaper — reasoning tokens count against
                max_output_tokens, and a reasoning model can burn the whole
                budget thinking and never emit a message. Can be flipped
                between jobs: `harness.reasoning = "on"`.
            workers: Concurrent requests to keep in flight. Set this to your
                LM Studio instance's parallel slot count (check the "parallel"
                field in /api/v1/models) for up to that much throughput.
                Each concurrent slot holds its own KV cache, so don't exceed
                what the server is configured for. Default 1 (serial).
                Results are still written in input order, so resume works
                identically at any worker count.
            chunk_chars: Character budget per map-reduce chunk in
                generate_long() (roughly 4 chars per token). The default
                16000 ≈ 4k tokens, sized for an 8k-token context with
                headroom for the system prompt and the response. Keep it
                below max_input_chars if both are set, or chunks will be
                truncated on their way out.
            validate_retries: When a `validator` is supplied to generate() or
                a processor, how many corrective attempts to make after an
                invalid output — the model is shown its rejected answer and
                the reason, and asked again. Still invalid after all
                attempts → GenerationError (an "[ERROR] ..." marker in batch
                mode, retried on resume like any other failure).
            compat_url: LM Studio's OpenAI-compat chat endpoint, used only
                for json_schema jobs (the v1 API doesn't support structured
                output). Derived from base_url's host by default.
        """
        self.base_url = base_url
        if compat_url is None:
            parts = urlsplit(base_url)
            compat_url = f"{parts.scheme}://{parts.netloc}/v1/chat/completions"
        self.compat_url = compat_url
        self.model = model
        self.timeout = (connect_timeout, timeout)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_output_tokens = max_output_tokens
        self.max_input_chars = max_input_chars
        self.reasoning = reasoning
        self.workers = max(1, workers)
        self.chunk_chars = chunk_chars
        self.validate_retries = validate_retries
        # Caps TOTAL in-flight requests at `workers`, even when parallel
        # loops nest (e.g. a directory job map-reducing each file). Held only
        # for the duration of each HTTP call, so it cannot deadlock.
        self._request_slots = threading.BoundedSemaphore(self.workers)
        # A Session reuses TCP connections across thousands of calls
        # instead of a fresh handshake per row.
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=max(10, self.workers))
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # ------------------------------------------------------------------ #
    # Core generation
    # ------------------------------------------------------------------ #

    def generate(self, system_prompt: str, input_text: str, temperature: float = 0.0,
                 validator=None, images=None, json_schema=None) -> str:
        """
        The core generation call. Enforces statelessness, strips reasoning
        blocks, and optionally validates the output.

        Args:
            validator: Optional callable applied to the output. Return
                None/True for valid; return False, a reason string, or raise
                ValueError for invalid. On invalid output the model is shown
                its rejected answer plus the reason and asked to correct it,
                up to `validate_retries` times. See the built-in `one_of()`
                and `valid_json()` validator factories.
            images: Optional list of images to attach (vision models only).
                Each entry is a file path or an existing "data:" URI; files
                are base64-encoded automatically. Requires the loaded model
                to report "vision": true in /api/v1/models.
            json_schema: Optional JSON Schema dict. When set, the request is
                routed through LM Studio's OpenAI-compat endpoint with
                grammar-constrained decoding — the output is GUARANTEED to be
                schema-conforming JSON (returned as a string). Stronger than
                a validator for structural correctness; combine with a
                validator only for semantic checks.

        Raises GenerationError after exhausting retries, instead of returning
        an error string that could silently poison downstream data.
        """
        output = self._dispatch(system_prompt, input_text, temperature, images, json_schema)
        if validator is None:
            return output
        return self._validate_and_correct(system_prompt, input_text, output, temperature,
                                          validator, images, json_schema)

    def _dispatch(self, system_prompt: str, input_text: str, temperature: float,
                  images=None, json_schema=None) -> str:
        """Route to the v1 endpoint, or the compat endpoint for schema jobs."""
        if self.max_input_chars is not None and len(input_text) > self.max_input_chars:
            print(f"  [warn] Input truncated from {len(input_text)} to {self.max_input_chars} chars.")
            input_text = input_text[: self.max_input_chars]
        if json_schema is not None:
            return self._request_schema(system_prompt, input_text, temperature, json_schema, images)
        return self._request(system_prompt, input_text, temperature, images)

    def _request(self, system_prompt: str, input_text: str, temperature: float,
                 images=None) -> str:
        """Single (retried) HTTP round-trip: payload build, POST, reasoning strip."""
        if images:
            # Multimodal form: the v1 API takes input as typed parts —
            # {"type": "text", "content": ...} / {"type": "image", "data_url": ...}
            payload_input = [{"type": "text", "content": input_text}]
            payload_input += [{"type": "image", "data_url": _to_data_url(img)} for img in images]
        else:
            payload_input = input_text

        payload = {
            "model": self.model,
            "system_prompt": system_prompt,
            "input": payload_input,
            "store": False,  # CRITICAL: Prevents LM Studio from accumulating background context memory
            "temperature": temperature,
        }
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning

        data = self._post_json(self.base_url, payload)

        # Iterate through the output array to extract only the final
        # payload, dropping the token-heavy reasoning scratchpad.
        for item in data.get("output", []):
            if item.get("type") == "message":
                return item.get("content", "").strip()

        # Diagnose the common failure: the model spent the entire
        # max_output_tokens budget on reasoning and never reached
        # the message block.
        stats = data.get("stats", {})
        reasoning_toks = stats.get("reasoning_output_tokens", 0)
        total_toks = stats.get("total_output_tokens", 0)
        if reasoning_toks and reasoning_toks >= total_toks - 2:
            raise GenerationError(
                f"Output cap exhausted by reasoning ({reasoning_toks}/{total_toks} tokens, "
                f"max_output_tokens={self.max_output_tokens}) — raise max_output_tokens "
                f"or set reasoning='off'."
            )
        raise GenerationError("No message block returned from API.")

    def _request_schema(self, system_prompt: str, input_text: str, temperature: float,
                        json_schema: dict, images=None) -> str:
        """
        Grammar-constrained generation via the OpenAI-compat endpoint —
        the only endpoint that supports response_format (the v1 API rejects
        it). Still stateless (no history), and the reasoning scratchpad
        arrives in a separate 'reasoning_content' field that is dropped.
        """
        if images:
            content = [{"type": "text", "text": input_text}]
            content += [{"type": "image_url", "image_url": {"url": _to_data_url(img)}} for img in images]
        else:
            content = input_text

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": content}],
            "temperature": temperature,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "output", "strict": True,
                                                "schema": json_schema}},
        }
        if self.max_output_tokens is not None:
            payload["max_tokens"] = self.max_output_tokens

        # Constrained decoding at temperature 0 can fall into a degenerate
        # repetition loop that runs until max_tokens truncates the JSON
        # mid-string. Deterministic retries would reproduce the exact same
        # loop, so failed attempts retry with a temperature bump.
        last_error, output = None, ""
        for attempt in range(self.max_retries):
            data = self._post_json(self.compat_url, payload)
            try:
                output = (data["choices"][0]["message"]["content"] or "").strip()
            except (KeyError, IndexError, TypeError) as e:
                raise GenerationError(f"Unexpected response shape from compat endpoint: {e}")
            try:
                json.loads(output)
                return output
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    payload["temperature"] = max(temperature, 0.4)
                    print(f"  [warn] Constrained output invalid (likely a repetition loop at "
                          f"temperature={temperature}) — retrying at temperature={payload['temperature']}.")
        raise GenerationError(
            f"Schema-constrained output is not valid JSON after {self.max_retries} attempts — "
            f"repetition loop or max_output_tokens={self.max_output_tokens} truncation "
            f"({last_error}) | output: {output[:200]!r}"
        )

    def _post_json(self, url: str, payload: dict) -> dict:
        """POST with retry/backoff, 4xx fail-fast, and the request-slot cap."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with self._request_slots:
                    response = self.session.post(url, json=payload, timeout=self.timeout)
                # 4xx means the request itself is wrong — retrying won't help.
                if 400 <= response.status_code < 500:
                    raise GenerationError(f"API rejected request ({response.status_code}): {response.text[:500]}")
                response.raise_for_status()
                return response.json()
            except GenerationError:
                raise
            except (requests.exceptions.RequestException, ValueError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_backoff * (2 ** attempt)
                    print(f"  [warn] Attempt {attempt + 1}/{self.max_retries} failed ({e}). Retrying in {delay:.0f}s...")
                    time.sleep(delay)

        raise GenerationError(f"API failed after {self.max_retries} attempts: {last_error}")

    def _validate_and_correct(self, system_prompt: str, input_text: str, output: str,
                              temperature: float, validator, images=None,
                              json_schema=None) -> str:
        """Run the validator; on failure, ask the model to correct itself."""
        problem = None
        for attempt in range(self.validate_retries + 1):
            problem = _run_validator(validator, output)
            if problem is None:
                return output
            if attempt < self.validate_retries:
                print(f"  [warn] Output failed validation ({problem}) — requesting a correction...")
                corrective_input = (
                    f"{input_text}\n\n---\n"
                    f"Your previous response was:\n{output}\n\n"
                    f"That response was rejected: {problem}\n"
                    f"Respond again, following the instruction exactly."
                )
                output = self._dispatch(system_prompt, corrective_input, temperature,
                                        images, json_schema)
        raise GenerationError(
            f"Validation failed after {self.validate_retries + 1} attempts: {problem} "
            f"| last output: {output[:200]!r}"
        )

    def _safe_generate(self, system_prompt: str, input_text: str, validator=None,
                       images=None, json_schema=None) -> str:
        """
        Batch-mode wrapper: converts hard failures into a greppable
        "[ERROR] ..." marker so one bad row doesn't abort a 10k-row run.
        """
        try:
            return self.generate(system_prompt, input_text, validator=validator,
                                 images=images, json_schema=json_schema)
        except GenerationError as e:
            print(f"  [error] {e}")
            return f"[ERROR] {e}"

    # ------------------------------------------------------------------ #
    # Map-reduce for documents longer than the context window
    # ------------------------------------------------------------------ #

    def generate_long(self, system_prompt: str, input_text: str, temperature: float = 0.0,
                      chunk_chars: Optional[int] = None, combine_prompt: Optional[str] = None,
                      validator=None, json_schema=None) -> str:
        """
        Like generate(), but handles inputs of any length via map-reduce:
        the text is split at natural boundaries into chunks of at most
        chunk_chars, the system prompt is applied to each chunk (in parallel,
        respecting `workers`), and the partial results are combined into one
        final answer. If the partials themselves exceed the chunk budget,
        they are combined hierarchically over multiple rounds.

        Short inputs fall through to a single plain generate() call, so this
        is always safe to use.

        Args:
            chunk_chars: Override the instance-level chunk budget.
            combine_prompt: Override the system prompt used for the combine
                step. Defaults to a generic "merge these partial results as
                if the instruction ran on the whole document" prompt built
                from system_prompt.
        """
        limit = chunk_chars or self.chunk_chars
        chunks = _split_text(input_text, limit)
        if len(chunks) == 1:
            return self.generate(system_prompt, input_text, temperature,
                                 validator=validator, json_schema=json_schema)

        print(f"  [map-reduce] Input is {len(input_text)} chars — mapping over {len(chunks)} chunks.")

        def run_chunk(pair):
            i, chunk = pair
            tagged = f"[Section {i + 1} of {len(chunks)} of a longer document]\n\n{chunk}"
            return self.generate(system_prompt, tagged, temperature)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            outputs = [o for _, o in self._ordered_parallel(pool, list(enumerate(chunks)), run_chunk)]

        reduce_sys = combine_prompt or (
            "You will be given several partial results. Each was produced by applying "
            "the following instruction to consecutive sections of a single long document:\n\n"
            f"--- INSTRUCTION ---\n{system_prompt}\n--- END INSTRUCTION ---\n\n"
            "Merge the partial results into one final result that satisfies the "
            "instruction as if it had been applied to the whole document at once. "
            "Do not mention sections, parts, or the merging process."
        )

        # Hierarchical reduce: combine partials in batches until everything
        # fits in one final combine call. Bounded rounds guard against
        # partials that refuse to shrink.
        # The validator / json_schema (if any) apply only to the FINAL result —
        # partials are intermediate scratch, not the deliverable format.
        last_joined = _join_partials(outputs)
        for _ in range(8):
            if len(outputs) == 1:
                if json_schema is not None:
                    # The lone partial wasn't schema-constrained; reformat it.
                    return self.generate(reduce_sys, outputs[0], temperature,
                                         validator=validator, json_schema=json_schema)
                if validator is None:
                    return outputs[0]
                return self._validate_and_correct(reduce_sys, last_joined, outputs[0],
                                                  temperature, validator)
            joined = _join_partials(outputs)
            if len(joined) <= limit:
                return self.generate(reduce_sys, joined, temperature,
                                     validator=validator, json_schema=json_schema)
            reduced = self._reduce_round(reduce_sys, outputs, limit, temperature)
            if reduced is None:  # no batching progress possible
                break
            print(f"  [map-reduce] Combining {len(outputs)} partials -> {len(reduced)}.")
            last_joined = joined
            outputs = reduced

        # Fallback: partials won't shrink under the chunk budget; do one
        # final oversized combine (max_input_chars truncation may apply).
        return self.generate(reduce_sys, _join_partials(outputs), temperature,
                             validator=validator, json_schema=json_schema)

    def _reduce_round(self, reduce_sys: str, outputs, limit: int, temperature: float):
        """Combine consecutive partials in batches that fit the chunk budget."""
        batches, current, cur_len = [], [], 0
        for o in outputs:
            if current and cur_len + len(o) > limit:
                batches.append(current)
                current, cur_len = [], 0
            current.append(o)
            cur_len += len(o) + 40  # + joining overhead
        if current:
            batches.append(current)
        if len(batches) >= len(outputs):
            return None

        def run_batch(pair):
            _i, batch = pair
            return self.generate(reduce_sys, _join_partials(batch), temperature)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return [o for _, o in self._ordered_parallel(pool, list(enumerate(batches)), run_batch)]

    def _safe_generate_long(self, system_prompt: str, input_text: str, validator=None,
                            json_schema=None) -> str:
        """Batch-mode wrapper for generate_long(); mirrors _safe_generate()."""
        try:
            return self.generate_long(system_prompt, input_text, validator=validator,
                                      json_schema=json_schema)
        except GenerationError as e:
            print(f"  [error] {e}")
            return f"[ERROR] {e}"

    def _ordered_parallel(self, pool, items, fn):
        """
        Run fn(item) across the thread pool while yielding (item, result) in
        input order. At most workers*2 requests are in flight at once, so
        memory stays bounded on huge datasets and the ordered incremental
        writes that resume depends on keep working at any worker count.
        """
        pending = deque()
        for item in items:
            pending.append((item, pool.submit(fn, item)))
            while len(pending) >= self.workers * 2:
                done_item, fut = pending.popleft()
                yield done_item, fut.result()
        while pending:
            done_item, fut = pending.popleft()
            yield done_item, fut.result()

    # ------------------------------------------------------------------ #
    # CSV
    # ------------------------------------------------------------------ #

    def process_csv(self, input_file: str, output_file: str, target_col: str,
                    system_prompt: str, resume: bool = True, validator=None,
                    json_schema=None):
        """
        Reads a CSV, processes a target column, and saves to a new CSV with an
        added 'llm_output' column — or one column per prompt when
        system_prompt is a dict (see the class docstring).

        Rows are flushed to disk as they complete. If the run is interrupted,
        calling again with resume=True (the default) skips the rows already
        present in the output file and appends from there. Pass resume=False
        to start over.
        """
        rows_done = 0
        if resume and os.path.exists(output_file):
            _repair_trailing_partial_line(output_file)
            with open(output_file, mode='r', encoding='utf-8', newline='') as f:
                rows_done = max(0, sum(1 for _ in csv.reader(f)) - 1)  # minus header

        mode = 'a' if rows_done > 0 else 'w'
        print(f"Starting CSV processing for: {input_file}"
              + (f" (resuming after {rows_done} completed rows)" if rows_done else ""))

        with open(input_file, mode='r', encoding='utf-8') as infile, \
             open(output_file, mode=mode, encoding='utf-8', newline='') as outfile:

            specs = _normalize_prompts(system_prompt, validator)
            reader = csv.DictReader(infile)
            fieldnames = list(reader.fieldnames or [])
            for key, _p, _v in specs:
                if key not in fieldnames:
                    fieldnames.append(key)
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            if mode == 'w':
                writer.writeheader()

            def rows_to_do():
                for row_idx, row in enumerate(reader, 1):
                    if row_idx > rows_done:
                        yield row_idx, row

            def run(pair):
                _idx, row = pair
                input_data = row.get(target_col, "")
                return {key: (self._safe_generate(prompt, input_data, vdt, json_schema=json_schema)
                              if input_data else "")
                        for key, prompt, vdt in specs}

            failures = 0
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for (row_idx, row), outputs in self._ordered_parallel(pool, rows_to_do(), run):
                    row.update(outputs)
                    failures += sum(1 for o in outputs.values() if o.startswith("[ERROR]"))
                    print(f"Row {row_idx} done.")
                    writer.writerow(row)
                    # Durable checkpoint: survive a crash/power-off mid-run.
                    outfile.flush()
                    os.fsync(outfile.fileno())

        print(f"CSV processing complete.{f' ({failures} outputs failed — grep for [ERROR])' if failures else ''}")

    # ------------------------------------------------------------------ #
    # JSON (array of objects)
    # ------------------------------------------------------------------ #

    def process_json_list(self, input_file: str, output_file: str, target_key: str,
                          system_prompt: str, resume: bool = True,
                          checkpoint_every: int = 1, retry_errors: bool = True,
                          validator=None, json_schema=None):
        """
        Iterates through an array of JSON objects, processes a target key, and
        saves the updated array.

        The output file is checkpointed (atomic temp-file swap) every
        `checkpoint_every` items, so an interruption loses at most that many
        results. On resume, output keys that already carry a non-error value
        are skipped; keys marked "[ERROR] ..." are retried unless
        retry_errors=False. In multi-prompt mode (system_prompt as a dict)
        only the missing/failed keys of each object are regenerated.

        Note: the whole array lives in memory. For datasets too large for
        that, use process_jsonl() instead.
        """
        source = input_file
        if resume and os.path.exists(output_file):
            source = output_file
            print(f"Resuming from existing output: {output_file}")

        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of objects.")

        specs = _normalize_prompts(system_prompt, validator)

        def _key_done(obj, key):
            out = obj.get(key)
            if out is None:
                return False
            if retry_errors and isinstance(out, str) and out.startswith("[ERROR]"):
                return False
            return True

        todo = []
        for i, obj in enumerate(data):
            if not obj.get(target_key, ""):
                continue
            pending = [(key, prompt, vdt) for key, prompt, vdt in specs if not _key_done(obj, key)]
            if pending:
                todo.append((i, obj, pending))
        print(f"Starting JSON processing for: {input_file} ({len(todo)} of {len(data)} objects to do)")

        def run(item):
            _i, obj, pending = item
            return {key: self._safe_generate(prompt, str(obj.get(target_key, "")), vdt,
                                             json_schema=json_schema)
                    for key, prompt, vdt in pending}

        dirty = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for (i, obj, _pending), outputs in self._ordered_parallel(pool, todo, run):
                if json_schema is not None:
                    outputs = {k: _parse_unless_error(v) for k, v in outputs.items()}
                obj.update(outputs)
                print(f"Object {i+1}/{len(data)} done.")
                dirty += 1
                if dirty >= checkpoint_every:
                    _atomic_json_dump(data, output_file)
                    dirty = 0

        _atomic_json_dump(data, output_file)
        print("JSON processing complete.")

    # ------------------------------------------------------------------ #
    # JSONL (streaming — constant memory)
    # ------------------------------------------------------------------ #

    def process_jsonl(self, input_file: str, output_file: str, target_key: str,
                      system_prompt: str, resume: bool = True, validator=None,
                      json_schema=None):
        """
        Streams a JSON-Lines file (one object per line): constant memory
        regardless of dataset size, with per-line durability. Preferred over
        process_json_list for large datasets.

        Resume works by line count: completed output lines are counted and
        that many input lines are skipped.
        """
        lines_done = 0
        if resume and os.path.exists(output_file):
            _repair_trailing_partial_line(output_file)
            with open(output_file, 'r', encoding='utf-8') as f:
                lines_done = sum(1 for line in f if line.strip())

        mode = 'a' if lines_done > 0 else 'w'
        print(f"Starting JSONL processing for: {input_file}"
              + (f" (resuming after {lines_done} completed lines)" if lines_done else ""))

        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, mode=mode, encoding='utf-8') as outfile:

            def records_to_do():
                # Count only non-blank lines so resume line-counting stays
                # aligned even if the input contains blank lines.
                record_idx = 0
                for line in infile:
                    if not line.strip():
                        continue
                    record_idx += 1
                    if record_idx > lines_done:
                        yield record_idx, json.loads(line)

            specs = _normalize_prompts(system_prompt, validator)

            def run(pair):
                _idx, obj = pair
                input_data = obj.get(target_key, "")
                if not input_data:
                    return None
                return {key: self._safe_generate(prompt, str(input_data), vdt,
                                                 json_schema=json_schema)
                        for key, prompt, vdt in specs}

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for (record_idx, obj), outputs in self._ordered_parallel(pool, records_to_do(), run):
                    if outputs is not None:
                        if json_schema is not None:
                            outputs = {k: _parse_unless_error(v) for k, v in outputs.items()}
                        obj.update(outputs)
                    print(f"Line {record_idx} done.")
                    outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    outfile.flush()
                    os.fsync(outfile.fileno())
        print("JSONL processing complete.")

    # ------------------------------------------------------------------ #
    # Directory of text files
    # ------------------------------------------------------------------ #

    def process_directory(self, input_dir: str, output_dir: str, system_prompt: str,
                          ext: str = ".txt", resume: bool = True, map_reduce: bool = True,
                          validator=None, json_schema=None):
        """
        Reads files one-by-one from a directory, processes the text, and saves
        to an output directory. With resume=True (the default), files whose
        output already exists are skipped, so an interrupted run picks up
        where it stopped.

        With map_reduce=True (the default), files longer than chunk_chars are
        processed via generate_long() — chunked, mapped, and combined —
        instead of being truncated. Short files behave identically either way.

        With a single prompt string, each output file is the raw model
        response (unchanged behavior). In multi-prompt mode (system_prompt as
        a dict), each output file is a JSON object with one key per prompt.
        """
        print(f"Starting directory processing for: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Sorted for a deterministic order — os.listdir() order is arbitrary,
        # which would make progress reports (and debugging) inconsistent.
        todo = []
        for filename in sorted(os.listdir(input_dir)):
            if not filename.endswith(ext):
                continue
            output_path = os.path.join(output_dir, f"processed_{filename}")
            if resume and os.path.exists(output_path):
                # Retry files whose previous attempt hard-failed.
                if not _has_failed_marker(output_path):
                    print(f"Skipping (already done): {filename}")
                    continue
                print(f"Retrying previously failed file: {filename}")
            todo.append(filename)

        specs = _normalize_prompts(system_prompt, validator)
        multi = isinstance(system_prompt, dict)

        def run(filename):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                text_content = f.read()
            gen = self._safe_generate_long if map_reduce else self._safe_generate
            outputs = {key: gen(prompt, text_content, vdt, json_schema=json_schema)
                       for key, prompt, vdt in specs}
            if multi:
                return json.dumps(outputs, indent=2, ensure_ascii=False)
            return outputs["llm_output"]

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for filename, result in self._ordered_parallel(pool, todo, run):
                output_path = os.path.join(output_dir, f"processed_{filename}")
                # Write to a temp file first so a crash mid-write never leaves
                # a half-finished output that resume would mistake for complete.
                tmp_path = output_path + ".tmp"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                os.replace(tmp_path, output_path)
                print(f"File done: {filename}")
        print("Directory processing complete.")

    # ------------------------------------------------------------------ #
    # Directory of images (vision models)
    # ------------------------------------------------------------------ #

    def process_image_directory(self, input_dir: str, output_dir: str, system_prompt: str,
                                exts=(".png", ".jpg", ".jpeg", ".webp"),
                                user_text: str = "Process this image according to your instructions.",
                                resume: bool = True, validator=None, json_schema=None):
        """
        Runs each image in a directory through the model (vision models
        only — the loaded model must report "vision": true in /api/v1/models)
        and writes one 'processed_<name>.txt' file per image.

        Same semantics as process_directory: resume skips completed files
        and retries failed ones, system_prompt accepts a dict for
        multi-prompt mode (output becomes a JSON object per image), and
        validators apply per prompt.

        Args:
            user_text: The text part sent alongside the image; the
                system_prompt carries the actual instructions.
        """
        print(f"Starting image directory processing for: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        todo = []
        for filename in sorted(os.listdir(input_dir)):
            if not filename.lower().endswith(tuple(e.lower() for e in exts)):
                continue
            output_path = os.path.join(output_dir, f"processed_{filename}.txt")
            if resume and os.path.exists(output_path):
                if not _has_failed_marker(output_path):
                    print(f"Skipping (already done): {filename}")
                    continue
                print(f"Retrying previously failed image: {filename}")
            todo.append(filename)

        specs = _normalize_prompts(system_prompt, validator)
        multi = isinstance(system_prompt, dict)

        def run(filename):
            image_path = os.path.join(input_dir, filename)
            outputs = {key: self._safe_generate(prompt, user_text, vdt, images=[image_path],
                                                json_schema=json_schema)
                       for key, prompt, vdt in specs}
            if multi:
                return json.dumps(outputs, indent=2, ensure_ascii=False)
            return outputs["llm_output"]

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for filename, result in self._ordered_parallel(pool, todo, run):
                output_path = os.path.join(output_dir, f"processed_{filename}.txt")
                tmp_path = output_path + ".tmp"
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                os.replace(tmp_path, output_path)
                print(f"Image done: {filename}")
        print("Image directory processing complete.")


# ---------------------------------------------------------------------- #
# Validators
# ---------------------------------------------------------------------- #

def one_of(*labels, case_sensitive: bool = False):
    """
    Validator factory: the output must be exactly one of the given labels
    (surrounding whitespace ignored). Use for classification jobs:

        harness.process_jsonl(..., validator=one_of("POSITIVE", "NEGATIVE", "MIXED"))
    """
    def check(output: str):
        value = output.strip()
        allowed = labels if case_sensitive else tuple(l.upper() for l in labels)
        if (value if case_sensitive else value.upper()) not in allowed:
            return f"output must be exactly one of {list(labels)}, got {output.strip()[:80]!r}"
    return check


def valid_json(require_keys=None):
    """
    Validator factory: the output must parse as JSON (markdown code fences
    are tolerated and stripped before parsing). Optionally require top-level
    keys:

        harness.process_csv(..., validator=valid_json(require_keys=["name", "amount"]))
    """
    def check(output: str):
        text = output.strip()
        if text.startswith("```"):
            text = text.strip("`\n")
            if text.startswith("json"):
                text = text[4:]
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            return f"output must be valid JSON ({e})"
        if require_keys:
            if not isinstance(obj, dict):
                return f"JSON output must be an object with keys {list(require_keys)}"
            missing = [k for k in require_keys if k not in obj]
            if missing:
                return f"JSON output is missing required keys: {missing}"
    return check


def _normalize_prompts(system_prompt, validator):
    """
    Normalize the single-prompt and multi-prompt forms into a list of
    (output_key, prompt, validator) triples.
    """
    if isinstance(system_prompt, dict):
        specs = []
        for key, spec in system_prompt.items():
            if isinstance(spec, (tuple, list)):
                specs.append((key, spec[0], spec[1]))
            else:
                specs.append((key, spec, validator))
        if not specs:
            raise ValueError("Multi-prompt dict must contain at least one prompt.")
        return specs
    return [("llm_output", system_prompt, validator)]


def _run_validator(validator, output: str):
    """Normalize validator conventions. Returns None if valid, else a reason string."""
    try:
        result = validator(output)
    except ValueError as e:
        return str(e) or "validator rejected the output"
    if result is None or result is True:
        return None
    if result is False:
        return "output failed validation"
    return str(result)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _split_text(text: str, limit: int, separators=("\n\n", "\n", ". ", " ")):
    """
    Split text into chunks of at most `limit` chars, preferring natural
    boundaries (paragraphs, then lines, sentences, words, then a hard cut).
    Lossless: ''.join(chunks) == text.
    """
    if len(text) <= limit:
        return [text]
    for si, sep in enumerate(separators):
        if sep not in text:
            continue
        parts = text.split(sep)
        units = [p + sep for p in parts[:-1]] + [parts[-1]]
        chunks, current = [], ""
        for unit in units:
            if len(unit) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(_split_text(unit, limit, separators[si + 1:]))
            elif len(current) + len(unit) <= limit:
                current += unit
            else:
                chunks.append(current)
                current = unit
        if current:
            chunks.append(current)
        return chunks
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def _join_partials(outputs):
    return "\n\n".join(
        f"[Partial result {i + 1} of {len(outputs)}]\n{o}" for i, o in enumerate(outputs)
    )


_IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
}


def _to_data_url(image) -> str:
    """Accepts a file path or an existing data: URI; returns a data: URI."""
    if isinstance(image, str) and image.startswith("data:"):
        return image
    ext = os.path.splitext(str(image))[1].lower()
    mime = _IMAGE_MIME.get(ext)
    if mime is None:
        raise GenerationError(f"Unsupported image type {ext!r} for {image!r} "
                              f"(supported: {sorted(_IMAGE_MIME)})")
    with open(image, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_unless_error(value):
    """
    For json_schema jobs writing to JSON-native formats: store the parsed
    object rather than a JSON string. "[ERROR] ..." markers stay as strings
    so resume can spot and retry them.
    """
    if isinstance(value, str) and not value.startswith("[ERROR]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _has_failed_marker(path: str) -> bool:
    """
    True if a directory-job output file records a failed attempt: either a
    raw "[ERROR] ..." body (single-prompt mode) or a JSON object with an
    "[ERROR] ..." value (multi-prompt mode).
    """
    with open(path, 'r', encoding='utf-8') as f:
        head = f.read(65536)
    if head.startswith("[ERROR]"):
        return True
    if head.lstrip().startswith("{"):
        try:
            obj = json.loads(head)
        except json.JSONDecodeError:
            return False
        return isinstance(obj, dict) and any(
            isinstance(v, str) and v.startswith("[ERROR]") for v in obj.values()
        )
    return False


def _atomic_json_dump(data, path: str):
    """Write JSON via temp file + rename so the output is never half-written."""
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _repair_trailing_partial_line(path: str):
    """
    If the process was killed mid-write, the last line of an append-mode
    output may be torn (no trailing newline). Truncate it so resume counting
    starts from the last complete record.
    """
    with open(path, 'rb+') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b'\n':
            return
        # Walk backwards to the last newline and cut there.
        pos = size - 1
        while pos > 0:
            f.seek(pos - 1)
            if f.read(1) == b'\n':
                break
            pos -= 1
        f.truncate(pos)
        print(f"  [warn] Repaired torn final line in {path} (truncated {size - pos} bytes).")
