3# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reusable, domain-agnostic pipeline that uses Claude models in an orchestrator/worker/compiler/evaluator
pattern to generate, validate, and iteratively refine a **standalone Python analysis script** — not to
write general-purpose application code. The pipeline itself (`pipeline.py`) never changes per use case;
only the domain config (`bioimage_config.py`, `trello_config.py`, etc.) and input data change.

## Commands

Dependency management is via **pixi**, not pip/requirements.txt (the README's `pip install -r
requirements.txt` is aspirational — no requirements.txt exists in the repo).

```bash
pixi install                    # install/sync the environment from pixi.toml/pixi.lock
pixi run python app.py          # run the pipeline (bioimage config by default)
pixi run python app.py --config trello        # run against the Trello domain config
pixi run python app.py --config bioimage --report <path> --data-dir <path> --output-dir ./outputs --max-iterations 5
```

Docker is required for the execution-validation step of the pipeline (not for running `app.py` itself):

```bash
docker build -t bia-analysis:latest .    # build the sandbox image used by bioimage_config.py
```

Without a running Docker daemon, `execute_script_in_docker` returns `None` and the pipeline skips the
run step, evaluating code quality only (see `validate_execution` in `pipeline.py`).

There is no test suite, linter config, or CI in this repo currently.

`ANTHROPIC_API_KEY` must be set (`.env` file, loaded via `python-dotenv`, or exported in the shell).

## Architecture

### Pipeline flow (`pipeline.py`)

```
Orchestrator (1 call)  →  Workers (parallel, 1 call per function)  →  Compiler+Execution loop  →  Requirements Evaluator
```

- **Orchestrator**: given the task report + input metadata, designs a minimal architecture — in practice
  just `load_data()` and `main()` (see `ORCHESTRATOR_PROMPT`). Returns an `<analysis>` block and a
  `<tasks>` list parsed by `parse_tasks()`.
- **Workers**: one parallel LLM call per task (`asyncio.gather` in `_call_worker`), each implementing a
  single function to spec with no helpers, no defensive try/except.
- **Compiler + execution loop** (`_run_one_design`): compiles worker output into one script
  (`compile_script`), then runs it in a sandboxed Docker container (`execute_script_in_docker`) and asks
  an LLM to validate execution (`validate_execution`). Retries up to `max_compile_attempts` (default 3),
  feeding the execution error back into the next compile attempt.
- **Requirements Evaluator**: only runs once execution passes; checks the script produced 5+ metrics and
  3+ non-zero-byte PNGs, using the *actual on-disk file listing*, not just what the code claims to write
  (`_format_artifacts` flags 0-byte files as suspect).

### Best-of-N outer loop (`generate_and_optimize`)

Each iteration fans out `designs_per_iteration` (default 3) fully independent design attempts in
parallel via `_run_one_design`, each writing its artifacts to its own
`outputs/artifacts/iter_N/design_M/` subdirectory to avoid clobbering. Candidates are ranked by
`_candidate_score` — a lexicographic tuple `(req_pass, exec_pass, valid_png_count)` — so a
requirements-passing design always wins, and ties are broken by how many valid PNGs were produced.

If no design in an iteration passes requirements, **every** failing design's feedback is appended to
`feedback_history` (not just the best one), and the full accumulated history is passed into the next
iteration's orchestrator prompt. This is deliberate: it stops the model from oscillating (fixing issue A
by reintroducing issue B) by making every orchestrator redesign aware of all prior dead ends.

The single best candidate seen across *all* iterations (`best_candidate`, via `record_candidate`) is
returned even if the loop exhausts `max_iterations` without a pass — not just whatever the final
iteration produced.

### Docker sandboxing (`execute_script_in_docker`)

LLM-generated code is untrusted and is executed with `DOCKER_SANDBOX_FLAGS`: no network, capped
memory/CPU, read-only root filesystem (with a `tmpfs` for `/tmp`, `HOME`, and `MPLCONFIGDIR` since
matplotlib/font caches need somewhere writable), dropped capabilities, non-root user, and a process
limit. Treat these flags as a security boundary — don't loosen them without good reason.

### Adding a new domain

The pipeline is retargeted entirely through `PipelineConfig` (`config.py`) — no changes to
`pipeline.py` are needed. A domain config module must provide:

- `orchestrator_model`, `worker_model`, `compiler_model`, `executor_evaluator_model`,
  `requirements_evaluator_model` — role-based model selection (see `bioimage_config.py` /
  `trello_config.py` for current model assignments: Opus 4.8 for architecture, Sonnet 5/Haiku 4.5 for
  compilation/implementation/evaluation).
- `docker_image` — must already exist locally (built from a `Dockerfile` with the domain's libraries
  pre-installed) and must match `available_libraries`, since the generated script is restricted to
  exactly what's installed in that image.
- `available_libraries`, `domain_notes` — free-text constraints injected into every worker/compiler
  prompt.
- `extract_input_metadata(data_dir) -> str` — scans the input directory and returns a description fed to
  the orchestrator (e.g. `bioimage_config.py` reads TIFF shape/channel metadata via `bioio.BioImage`;
  `trello_config.py` summarizes JSON export structure).

Then wire the new config into `app.py`'s `--config` choices. Note: `trello_config.py` currently
references a `python-analysis:latest` Docker image that this repo's `Dockerfile` does not build (only
`bia-analysis:latest` is defined) — building that image is a prerequisite for the trello config to
validate execution.

### Structured I/O convention

All LLM prompts/responses use XML tags (`<analysis>`, `<tasks>`, `<task>`, `<response>`, `<evaluation>`,
`<feedback>`) parsed via `extract_xml()` / `parse_tasks()` in `pipeline.py`, with regex-based fallbacks
if strict XML parsing fails (tolerating minor formatting drift from the model). When editing prompts,
preserve these tags — downstream parsing depends on them.

### Generated-script conventions (enforced via prompts, not code)

Every compiled script is required (per `COMPILER_PROMPT`) to start with `# -*- coding: utf-8 -*-` and
have `main()` call `sys.stdout.reconfigure(encoding='utf-8')` as its first line, so UTF-8 output (emoji,
special characters) is safe across platforms inside the Docker container.