"""Generic multi-agent code-generation pipeline.

Orchestrator → Workers (parallel) → Compiler → Evaluator, with feedback loop.
Agnostic to domain — configure via PipelineConfig.
"""

import asyncio
import os
import random
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from anthropic import AsyncAnthropic
from config import PipelineConfig
from dotenv import load_dotenv
from pathlib import Path
from prompts import *

load_dotenv(override=True)
async_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Caps concurrent in-flight LLM requests across the whole pipeline (orchestrators, workers,
# compilers, evaluators all funnel through llm_call). Without this, designs_per_iteration
# parallel designs x per-design worker fan-out can easily put 15-20+ requests in flight at
# once, tripping rate limits. Override via LLM_MAX_CONCURRENCY.
LLM_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("LLM_MAX_CONCURRENCY", "8")))

# Core LLM interface
async def llm_call(prompt: str, system_prompt: str = None, model: str = None, cache_prompt: bool = False,
                   max_tokens: int = 8192) -> str:
    """
    Calls the model with the given prompt and returns the response.

    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str, optional): The system prompt.
        model (str, optional): The model to use for the call.
        cache_prompt (bool): Enable prompt caching for this call.
        max_tokens (int): Maximum tokens in response (default 8192).

    Returns:
        str: The response from the language model.
    """
    if model is None:
        raise ValueError("model must be provided")

    system_content = system_prompt
    if cache_prompt:
        system_content = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]

    messages = [{"role": "user", "content": prompt}]

    # These models use adaptive thinking; if max_tokens is exhausted during the
    # thinking phase the response comes back with a thinking block but no text.
    # Retry once with a larger budget before giving up.
    async with LLM_SEMAPHORE:
        for attempt, tokens in enumerate((max_tokens, max_tokens * 2)):
            response = await async_client.messages.create(
                model=model,
                max_tokens=tokens,
                system=system_content,
                messages=messages,
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            if text.strip():
                return text

            # No text produced. If we ran out of tokens (likely during thinking), retry bigger.
            if response.stop_reason != "max_tokens":
                break

    content_types = [block.type for block in response.content]
    raise ValueError(
        f"No text content in response (stop_reason={response.stop_reason}, "
        f"blocks={content_types}). The token budget was likely consumed by thinking; "
        f"try a larger max_tokens."
    )


# Helper functions for data extraction and processing
def extract_xml(text: str, tag: str) -> str:
    """Extracts the content of the specified XML tag from the given text (case-insensitive)."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else ""


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template, raising a clear error if a variable is missing."""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required prompt variable: {e}") from e


# Matches a bare `&` that isn't the start of a real XML entity/char reference - the model
# frequently writes plain prose (e.g. "cards & checklists") into <description> text, which is
# invalid XML and otherwise breaks the whole <tasks> block for a single stray character.
_BARE_AMPERSAND = re.compile(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)')


def parse_tasks(tasks_xml: str) -> list[dict]:
    """Parse XML tasks into a list of task dictionaries."""
    tasks = []
    sanitized = _BARE_AMPERSAND.sub('&amp;', tasks_xml)
    try:
        root = ET.fromstring(f"<root>{sanitized}</root>")
        task_elems = root.findall("task")

        for task_elem in task_elems:
            task = {}
            for child in task_elem:
                if child.text:
                    task[child.tag] = child.text.strip()
            if task:
                tasks.append(task)
    except ET.ParseError as e:
        print(f"Warning: Failed to parse tasks XML: {e}")
        print(f"DEBUG: Raw tasks_xml (first 500 chars):\n{tasks_xml[:500]}")
        # Fallback: try to extract tasks manually using regex (covers structural breakage that
        # ampersand-escaping alone can't fix, e.g. a stray literal '<' in a description).
        task_pattern = r'<task>(.*?)</task>'
        for match in re.finditer(task_pattern, tasks_xml, re.DOTALL):
            task_content = match.group(1)
            task = {}
            for field in ("function", "description", "input", "output"):
                field_match = re.search(f'<{field}>(.*?)</{field}>', task_content, re.DOTALL)
                if field_match:
                    task[field] = field_match.group(1).strip()
            if task:
                tasks.append(task)
    return tasks


# Sandbox flags for running untrusted, LLM-generated code. Docker here provides both
# dependency pinning AND isolation. Tune these if a host/platform rejects a flag.
DOCKER_SANDBOX_FLAGS = [
    "--network", "none",  # no network access
    "--memory", "1g",  # cap RAM
    "--memory-swap", "1g",  # == memory, so swap is disabled
    "--cpus", "2",  # cap CPU
    "--pids-limit", "256",  # limit processes (fork-bomb guard)
    "--read-only",  # read-only root filesystem
    "--cap-drop", "ALL",  # drop all Linux capabilities
    "--security-opt", "no-new-privileges",  # block privilege escalation
    "--user", "1000:1000",  # run as non-root
    # Writable scratch for the non-root user under a read-only root (matplotlib/font cache, etc.)
    "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
]


def execute_script_in_docker(script: str, data_dir: str, docker_image: str, timeout: int = 300,
                             artifacts_dir: str = None) -> tuple[bool, str, list[dict]]:
    """
    Execute script in a sandboxed Docker container to verify it works and capture produced files.
    Returns (success, output_or_error, artifacts) or (None, message, []) if Docker unavailable.
    Each artifact is a dict: {"name": str, "size": int}. Files are copied to artifacts_dir if given.
    """
    try:
        subprocess.run(["docker", "ps"], capture_output=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, "Docker not available - skipping execution test", []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "script.py"
            script_path.write_text(script, encoding="utf-8")

            docker_cmd = [
                "docker", "run", "--rm",
                *DOCKER_SANDBOX_FLAGS,
                "-v", f"{Path(data_dir).absolute()}:/data:ro",
                "-v", f"{tmpdir}:/work",
                "-w", "/work",
                "-e", "INPUT_FOLDER=/data",
                # Point HOME and matplotlib's cache at the writable tmpfs (root fs is read-only)
                "-e", "HOME=/tmp",
                "-e", "MPLCONFIGDIR=/tmp/mpl",
                docker_image,
                "python", "script.py"
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=timeout + 30,
                text=True
            )

            # List files the script produced in /work (everything except the script itself),
            # while the temp dir still exists, and persist them so they survive cleanup.
            # Clear the artifacts dir first so it only ever reflects the latest run.
            if artifacts_dir:
                adir = Path(artifacts_dir)
                adir.mkdir(parents=True, exist_ok=True)
                for stale in adir.iterdir():
                    if stale.is_file():
                        stale.unlink()

            artifacts = []
            for produced in sorted(Path(tmpdir).iterdir()):
                if produced.name == "script.py" or not produced.is_file():
                    continue
                artifacts.append({"name": produced.name, "size": produced.stat().st_size})
                if artifacts_dir:
                    shutil.copy2(produced, Path(artifacts_dir) / produced.name)

            if result.returncode == 0:
                return True, result.stdout or "Script executed successfully", artifacts
            else:
                return False, result.stderr or "Script execution failed with no error output", artifacts

    except subprocess.TimeoutExpired:
        return False, f"Script execution timed out (>{timeout}s)", []
    except Exception as e:
        if "daemon" in str(e).lower() or "pipe" in str(e).lower():
            return None, "Docker daemon not running - skipping execution test", []
        return False, f"Execution error: {str(e)}", []


# Core async functions for the compilation pipeline
async def compile_script(orchestrator_results: dict, config: PipelineConfig, error_feedback: str = "",
                         seed_script: str = None) -> str:
    """Compile worker functions into a single executable script, optionally fixing a prior execution
    error and/or improving a seed_script (a prior working script this design is mutating) instead of
    assembling from scratch."""
    analysis = orchestrator_results["analysis"]

    functions_text = "\n\n".join([
        f"# Function: {result['function']}\n# Description: {result['description']}\n{result['result']}"
        for result in orchestrator_results["worker_results"]
    ])

    if not functions_text.strip():
        print("WARNING: No worker functions were generated!")

    error_section = ""
    if error_feedback:
        error_section = (
            f"\nThe PREVIOUS compilation FAILED to execute. Fix this error in your output:\n"
            f"{error_feedback}\n"
        )

    seed_section = ""
    if seed_script:
        seed_section = (
            "\nSEED SCRIPT (the working script this design is improving upon):\n"
            f"{seed_script}\n\n"
            "Integrate the functions above into an IMPROVED version of this seed script - carry over "
            "parts of the seed that still apply, replace or extend the parts the new/changed functions "
            "address, and remove anything superseded. Do not discard working seed logic that the "
            "architecture and functions above don't touch.\n"
        )

    compiler_input = COMPILER_PROMPT.format(
        analysis=analysis,
        functions=functions_text,
        library_notes=config.available_libraries,
        error_feedback=error_section,
        seed_section=seed_section,
    )

    compiled_response = await llm_call(compiler_input, system_prompt=COMPILER_SYSTEM, model=config.compiler_model,
                                       cache_prompt=True, max_tokens=16384)
    compiled_script = extract_xml(compiled_response, "response")

    if not compiled_script.strip():
        # If no response tag found, extract by finding Python code block
        lines = compiled_response.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith("<") and not line.strip().startswith(">"):
                start_idx = i
                break
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() and not lines[i].strip().startswith("<"):
                end_idx = i + 1
                break
        compiled_script = "\n".join(lines[start_idx:end_idx])

    # Strip markdown code block markers if present
    compiled_script = compiled_script.strip()
    if compiled_script.startswith("```"):
        lines = compiled_script.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        compiled_script = "\n".join(lines).strip()

    return compiled_script


def validate_execution(compiled_script: str, config: PipelineConfig, data_dir: str = None,
                       artifacts_dir: str = None) -> tuple[str, str, str, list[dict]]:
    """Check if script executes. Grounded directly in the Docker exit code - no LLM judgment.

    Returns (PASS/FAIL/SKIPPED, feedback, execution_output, artifacts). SKIPPED means execution
    was never actually attempted (no data_dir, or Docker unavailable): this must never be reported
    as PASS, since nothing was verified to run.
    """
    if not data_dir:
        return "SKIPPED", "No data directory provided - execution was not verified.", "", []

    exec_success, exec_output, artifacts = execute_script_in_docker(
        compiled_script, data_dir, config.docker_image, artifacts_dir=artifacts_dir)

    if exec_success is None:
        return "SKIPPED", f"Docker unavailable - execution was not verified: {exec_output}", exec_output, artifacts
    if exec_success:
        return "PASS", "Script executed successfully.", exec_output, artifacts
    # Keep the TAIL: Python puts the actual exception last, after the traceback frames
    return "FAIL", f"Script execution failed:\n{exec_output[-2000:]}", exec_output, artifacts


def _format_artifacts(artifacts: list[dict]) -> str:
    """Render the list of produced files with sizes; flag empty files as suspect."""
    if not artifacts:
        return "(No files were produced by the script.)"
    lines = []
    for a in artifacts:
        flag = "  [WARNING: 0 bytes - likely not a valid image]" if a["size"] == 0 else ""
        lines.append(f"- {a['name']} ({a['size']} bytes){flag}")
    return "\n".join(lines)


_CRITERION_PATTERN = re.compile(r'<criterion\s+met="(true|false)"\s*/?>', re.IGNORECASE)


async def validate_requirements(compiled_script: str, report: str, criteria: str, exec_output: str,
                                config: PipelineConfig, artifacts: list[dict] = None) -> tuple[float, bool, str]:
    """Check the script's actual output against each bullet of the extracted success criteria.

    Returns (req_score, req_pass, feedback): req_score is met/total across every <criterion> tag the
    validator emitted (0.0 if it emitted none - treated as a full miss, not a free pass); req_pass is
    True only when every criterion was met. The graded score, not just the boolean, is what lets a
    mutated design's fitness be compared even when neither pass outright.
    """
    artifacts_listing = _format_artifacts(artifacts or [])
    validator_input = REQUIREMENTS_VALIDATOR_PROMPT.format(
        report=report,
        criteria=criteria,
        content=compiled_script,
        # Keep the TAIL: the script prints metrics then data-gap suggestions at the very end
        execution_result=f"Console output:\n{exec_output[-3000:]}\n\nFiles actually produced on disk:\n{artifacts_listing}"
    )

    validator_response = await llm_call(validator_input, system_prompt=EVALUATOR_SYSTEM,
                                        model=config.requirements_evaluator_model, cache_prompt=True)
    verdicts = _CRITERION_PATTERN.findall(validator_response)
    feedback = extract_xml(validator_response, "feedback").strip()

    if not verdicts:
        print(
            f"DEBUG: Requirements validator emitted no <criterion> tags (first 800 chars):\n{validator_response[:800]}")
        if not feedback:
            feedback = validator_response.strip()

    total = len(verdicts)
    met = sum(1 for v in verdicts if v.lower() == "true")
    req_score = met / total if total else 0.0
    req_pass = total > 0 and met == total

    return req_score, req_pass, feedback


async def _call_worker(task_info: dict, task_index: int, report: str, input_metadata: str,
                       config: PipelineConfig) -> dict:
    """Call worker for a single task. Used for parallel execution."""
    func_name = task_info.get("function", f"task_{task_index}")
    worker_input = format_prompt(
        WORKER_PROMPT,
        original_report=report,
        function=func_name,
        description=task_info.get("description", ""),
        input=task_info.get("input", ""),
        output=task_info.get("output", ""),
        input_data=input_metadata,
        library_notes=config.available_libraries,
        domain_notes=config.domain_notes,
    )
    worker_response = await llm_call(worker_input, system_prompt=WORKER_SYSTEM, model=config.worker_model,
                                     cache_prompt=True)
    worker_content = extract_xml(worker_response, "response")
    return {
        "function": func_name,
        "description": task_info.get("description", ""),
        "result": worker_content,
    }


def _candidate_score(candidate: dict) -> tuple:
    """Rank candidates lexicographically: execution-pass first, then the graded requirements score.

    exec_pass is the high-order bit because it's the hard, Docker-grounded oracle signal - nothing
    the noisy LLM requirements judgment says should ever outrank it. req_score (met/total against the
    extracted rubric, from validate_requirements) only ever separates designs that already agree on
    exec_pass, and unlike a boolean req_pass it gives a real gradient both below and approaching the
    pass line - which is what lets mutation-from-seed (see generate_and_optimize) tell a design that
    got closer from one that didn't.
    """
    return (candidate["exec_pass"], candidate.get("req_score", 0.0))


def pick_best_seed(archive: list[dict]) -> dict | None:
    """Return the highest-scoring node in the archive, or None if it's empty."""
    if not archive:
        return None
    return max(archive, key=lambda node: node["score"])


def pick_other_seed(archive: list[dict], exclude: dict) -> dict | None:
    """Return a different archived node than `exclude`, chosen at random for diversity.

    None if the archive is empty or `exclude` is the only node in it.
    """
    others = [node for node in archive if node is not exclude]
    if not others:
        return None
    return random.choice(others)


_TOKEN_PATTERN = re.compile(r'[a-z0-9]+')


def _token_set(text: str) -> set:
    """Lowercase, split on non-alphanumerics, drop tokens under 3 chars."""
    return {t for t in _TOKEN_PATTERN.findall(text.lower()) if len(t) >= 3}


def _jaccard(a: set, b: set) -> float:
    """Token-set Jaccard similarity. Two empty sets score 0.0 - no text means no evidence of
    similarity, not a false "identical designs" signal."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _log_iteration_diversity(results: list[dict], iteration: int) -> None:
    """Measurement only - log pairwise token-set Jaccard similarity across designs' <analysis> text.
    Does not affect selection, feedback, or fan-out. Candidates without an "analysis" key (e.g. a
    design that raised an exception before the orchestrator call completed) are skipped.
    """
    designs = [(c["label"], _token_set(c["analysis"])) for c in results if "analysis" in c]
    pairs = [
        (designs[i][0], designs[j][0], _jaccard(designs[i][1], designs[j][1]))
        for i in range(len(designs))
        for j in range(i + 1, len(designs))
    ]

    if not pairs:
        print(f"[diversity] iteration {iteration}: mean=n/a (fewer than 2 analyses to compare)")
        return

    mean_similarity = sum(sim for _, _, sim in pairs) / len(pairs)
    pair_str = ", ".join(f"{a}~{b}={sim:.2f}" for a, b, sim in pairs)
    print(f"[diversity] iteration {iteration}: mean={mean_similarity:.2f}  pairs: {pair_str}")


def _journal_entry(candidate: dict, iteration: int) -> str:
    """One line recording what a design tried and what happened - not just failures.

    Plain structured text, no LLM summarization: label, a one-line approach summary (the first
    line of its <analysis>), the exec/requirements outcome, how many real artifacts it produced,
    and - only if it failed - the specific reason from its feedback.
    """
    label = candidate.get("label", "?")
    analysis = (candidate.get("analysis") or "").strip()
    approach = analysis.splitlines()[0].strip() if analysis else "(no analysis available)"
    if len(approach) > 200:
        approach = approach[:200].rstrip() + "..."

    exec_verdict = candidate.get("exec_verdict", "ERROR")
    req_pass = candidate.get("req_pass", False)
    if exec_verdict == "PASS":
        req_status = f"{'PASS' if req_pass else 'FAIL'} (score={candidate.get('req_score', 0.0):.2f})"
    else:
        req_status = "n/a"
    valid_artifacts = sum(1 for a in candidate.get("artifacts") or [] if a.get("size", 0) > 0)

    entry = (
        f"[Iteration {iteration}] {label}: approach=\"{approach}\" | "
        f"exec={exec_verdict} req={req_status} | artifacts={valid_artifacts}"
    )
    if not req_pass and candidate.get("feedback"):
        entry += f" | reason: {candidate['feedback']}"
    return entry


async def _run_one_design(report: str, criteria: str, input_metadata: str, config: PipelineConfig, data_dir: str,
                          feedback_section: str, stance: str, artifacts_dir: str, label: str,
                          max_compile_attempts: int = 3, seed_script: str = None,
                          seed_label: str = None) -> dict:
    """Run one full design attempt (orchestrate → workers → compile/execute loop → requirements).

    If seed_script is given (a prior candidate's working script, e.g. from the archive), the
    orchestrator and compiler are instructed to IMPROVE it rather than design from scratch - a
    mutation, not a diff/patch. Safe because the Docker oracle in the compile/execute loop below
    still catches any regression the mutation introduces, exactly as it would for a from-scratch
    design. seed_label is purely for logging (which archived node this design mutated).

    Returns a candidate dict: {script, exec_pass, req_pass, artifacts, artifacts_dir, feedback, label,
    analysis}. `feedback` is empty on full pass, else a description of what failed (for the redesign
    history). `analysis` is the orchestrator's raw <analysis> text ("" if never produced).
    """

    def log(msg):
        print(f"  [{label}] {msg}")

    log(f"Seed: mutating {seed_label}" if seed_script else "Seed: none (from scratch)")

    orchestrator_seed_section = ""
    if seed_script:
        orchestrator_seed_section = (
            "\nSEED SCRIPT (a working script from a prior design that already executed successfully):\n"
            f"{seed_script}\n\n"
            "Your job is to IMPROVE this script so it better satisfies the Success Criteria and the "
            "journal above - not to design a new architecture from scratch. Keep what already works; "
            "change only what's needed to fix known issues or satisfy criteria the seed doesn't yet "
            "meet.\n"
        )

    # ORCHESTRATOR: design the architecture
    orchestrator_input = format_prompt(
        ORCHESTRATOR_PROMPT, report=report, criteria=criteria, input_data=input_metadata,
        feedback=feedback_section, stance=stance, seed_section=orchestrator_seed_section,
    )
    orchestrator_response = await llm_call(orchestrator_input, system_prompt=ORCHESTRATOR_SYSTEM,
                                           model=config.orchestrator_model, cache_prompt=True)
    analysis = extract_xml(orchestrator_response, "analysis").strip()
    tasks = parse_tasks(extract_xml(orchestrator_response, "tasks"))
    log(f"Architecture: {len(tasks)} functions")

    # WORKERS: implement each function in parallel
    worker_results = await asyncio.gather(
        *[_call_worker(t, i, report, input_metadata, config) for i, t in enumerate(tasks, 1)]
    )
    orchestrator_results = {"analysis": analysis, "worker_results": worker_results}

    # INNER LOOP: Compiler + (grounded) Execution check
    compiled_script, exec_output, artifacts = None, "", []
    execution_passed = False
    exec_verdict = "FAIL"
    compile_error = ""
    for attempt in range(max_compile_attempts):
        log(f"Compile attempt {attempt + 1}/{max_compile_attempts}...")
        compiled_script = await compile_script(orchestrator_results, config, error_feedback=compile_error,
                                               seed_script=seed_script)
        exec_verdict, exec_feedback, exec_output, artifacts = validate_execution(
            compiled_script, config, data_dir, artifacts_dir=artifacts_dir)
        log(f"Execution: {exec_verdict}")
        # SKIPPED (no Docker) is terminal too - there's no error to fix, so retrying compiles
        # the same script again for nothing. It is NOT the same as a verified PASS though.
        if exec_verdict in ("PASS", "SKIPPED"):
            execution_passed = True
            break
        if attempt < max_compile_attempts - 1:
            compile_error = exec_feedback

    if not execution_passed:
        log(f"[FAILED] Did not execute after {max_compile_attempts} attempts.")
        return {
            "script": compiled_script, "exec_pass": False, "req_pass": False, "req_score": 0.0,
            "artifacts": artifacts, "artifacts_dir": artifacts_dir, "label": label, "analysis": analysis,
            "exec_verdict": "FAIL",
            "feedback": f"Execution failed after {max_compile_attempts} compile attempts: {exec_feedback}",
        }

    if exec_verdict == "SKIPPED":
        # Execution was never verified (no data_dir / Docker unavailable), so there are no real
        # artifacts to grade - a requirements call here is a guaranteed-FAIL judge call paid for
        # nothing. Short-circuit instead of spending one per design per iteration.
        log("Requirements: SKIPPED (execution unverified, skipping judge call)")
        return {
            "script": compiled_script, "exec_pass": False, "req_pass": False, "req_score": 0.0,
            "artifacts": artifacts, "artifacts_dir": artifacts_dir, "label": label, "analysis": analysis,
            "exec_verdict": "SKIPPED",
            "feedback": f"Execution was not verified, so requirements cannot be checked: {exec_feedback}",
        }

    # REQUIREMENTS VALIDATOR: only reached on a verified execution PASS (FAIL returned above,
    # SKIPPED short-circuited above).
    req_score, req_passed, req_feedback = await validate_requirements(
        compiled_script, report, criteria, exec_output, config, artifacts=artifacts)
    log(f"Requirements: {'PASS' if req_passed else 'FAIL'} (score={req_score:.2f})")
    return {
        "script": compiled_script, "exec_pass": True, "req_pass": req_passed, "req_score": req_score,
        "artifacts": artifacts, "artifacts_dir": artifacts_dir, "label": label, "analysis": analysis,
        "exec_verdict": "PASS",
        "feedback": "" if req_passed else f"Executed cleanly but requirements not met: {req_feedback}",
    }


async def generate_and_optimize(report: str, config: PipelineConfig, data_dir: str = None,
                                max_iterations: int = 2, output_dir: str = None,
                                designs_per_iteration: int = 3) -> str:
    """Best-of-N loop: each iteration fans out N independent designs, keeps the best, redesigns.

    Set designs_per_iteration=1 for the classic single-design-per-iteration behavior.
    """
    # Accumulated failure history so each orchestrator redesign sees ALL prior issues,
    # not just the most recent one (prevents fix-A-breaks-B oscillation).
    feedback_history = []
    # Best script seen so far, kept across iterations so a timeout returns the best
    # candidate (not merely whatever the last iteration produced).
    best_candidate = None
    # Flat scored archive of every candidate that has executed successfully, across all
    # iterations - reusable seed nodes for future design generation. This is NOT a tree search:
    # no parent pointers, no node-expansion graph, no search controller - just a capped,
    # score-ranked pool that pick_best_seed()/pick_other_seed() draw from.
    archive: list[dict] = []
    input_metadata = config.extract_input_metadata(data_dir) if data_dir else "(No input data provided)"

    # Parse the report into a success rubric ONCE, shared by every design/iteration. This is what
    # actually makes the pipeline domain-agnostic: without it, the orchestrator and requirements
    # validator prompts would have to hardcode the shape of "success" (metric counts, plot counts,
    # file types) for one specific kind of report, pre-deciding every axis a design could vary on.
    # If extraction itself fails (e.g. a transient rate-limit error), fall back to the raw report
    # instead of leaving every design this run pointed at an empty rubric.
    try:
        criteria_input = format_prompt(CRITERIA_PROMPT, report=report, input_data=input_metadata)
        criteria_response = await llm_call(criteria_input, system_prompt=CRITERIA_SYSTEM,
                                           model=config.requirements_evaluator_model, cache_prompt=True)
        criteria = extract_xml(criteria_response, "criteria").strip() or criteria_response.strip()
    except Exception as e:
        print(f"WARNING: Criteria extraction failed ({e!r}); falling back to the raw report as the criteria.")
        criteria = report
    print(f"\nSuccess criteria extracted from report:\n{criteria}\n")

    # Base dir under which each design gets its OWN iter_N/design_M subdir, so the files
    # on disk always match whichever script we ultimately return.
    artifacts_base = str(Path(output_dir) / "artifacts") if output_dir else None

    def record_candidate(candidate: dict, iteration: int):
        """Keep the highest-scoring candidate across all iterations; ties resolve to the earliest."""
        nonlocal best_candidate
        candidate = {**candidate, "iteration": iteration}
        if best_candidate is None or _candidate_score(candidate) > _candidate_score(best_candidate):
            best_candidate = candidate

    def update_archive(candidate: dict, iteration: int):
        """Add an executed candidate to the archive, keeping only the top 5 by score."""
        nonlocal archive
        if not candidate.get("exec_pass"):
            return
        archive.append({**candidate, "score": _candidate_score(candidate), "iteration": iteration})
        archive.sort(key=lambda node: node["score"], reverse=True)
        del archive[5:]

    def print_artifacts(candidate: dict):
        """Print the produced-file listing for a candidate, if any."""
        if candidate["artifacts"] and candidate["artifacts_dir"]:
            print(f"Produced {len(candidate['artifacts'])} file(s) in: {candidate['artifacts_dir']}")
            for a in candidate["artifacts"]:
                print(f"  - {a['name']} ({a['size']} bytes)")

    for iteration in range(max_iterations):
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration + 1}/{max_iterations}  ({designs_per_iteration} parallel designs)")
        print(f"{'=' * 80}")

        if feedback_history:
            print(f"\nRedesigning based on accumulated journal...")
            joined = "\n\n".join(feedback_history)
            feedback_section = (
                "Journal of every design tried so far, what happened, and why (not just a list of "
                "complaints) - use it to build on what worked and avoid repeating what didn't. "
                "Address ALL known issues at once; do NOT reintroduce an earlier problem while "
                "fixing a later one:\n"
                f"{joined}"
            )
        else:
            feedback_section = ""

        # Fan out N independent designs, each isolated in its own artifacts subdir. Stances cycle
        # (design m gets design_stances[m % len]) so extra designs beyond the stance list length
        # just repeat stances rather than erroring - no LLM call needed to pick them.
        design_dirs = [
            str(Path(artifacts_base) / f"iter_{iteration + 1}" / f"design_{m + 1}") if artifacts_base else None
            for m in range(designs_per_iteration)
        ]
        labels = [f"I{iteration + 1}.D{m + 1}" for m in range(designs_per_iteration)]
        stances = [config.design_stances[m % len(config.design_stances)] for m in range(designs_per_iteration)]

        # Seed some designs from the archive once it's non-empty, mutating a prior working script
        # instead of designing from scratch - orthogonal to the stance assignment above. Design 0
        # mutates the best archived node, design 1 mutates a different node (for diversity), any
        # remaining designs stay from-scratch (exploration). Iteration 1's archive is empty, so
        # every design that iteration is from-scratch. A mutated design that regresses simply loses
        # on _candidate_score and never overwrites a better archived node - the Docker oracle in the
        # compile/execute loop still catches any regression the mutation introduces.
        seed_scripts = [None] * designs_per_iteration
        seed_labels = [None] * designs_per_iteration
        if archive:
            best_seed = pick_best_seed(archive)
            if designs_per_iteration >= 1:
                seed_scripts[0], seed_labels[0] = best_seed["script"], best_seed["label"]
            if designs_per_iteration >= 2:
                other_seed = pick_other_seed(archive, exclude=best_seed)
                if other_seed:
                    seed_scripts[1], seed_labels[1] = other_seed["script"], other_seed["label"]

        raw_results = await asyncio.gather(*[
            _run_one_design(report, criteria, input_metadata, config, data_dir, feedback_section,
                            stances[m], design_dirs[m], label=labels[m],
                            seed_script=seed_scripts[m], seed_label=seed_labels[m])
            for m in range(designs_per_iteration)
        ], return_exceptions=True)

        # A design that raised (e.g. a rate_limit_error) scores as a zero candidate instead
        # of killing the whole iteration - the other parallel designs still get a chance.
        results = []
        for label, result in zip(labels, raw_results):
            if isinstance(result, BaseException):
                print(f"  [{label}] [ERROR] {result!r}")
                result = {
                    "script": None, "exec_pass": False, "req_pass": False, "req_score": 0.0,
                    "artifacts": [], "artifacts_dir": None, "label": label, "exec_verdict": "ERROR",
                    "feedback": f"Design raised an exception before completing: {result!r}",
                }
            results.append(result)

        _log_iteration_diversity(results, iteration + 1)

        # Score every design and update the global best.
        for candidate in results:
            record_candidate(candidate, iteration + 1)

        # Archive every executed design as a reusable seed node for future design generation.
        for candidate in results:
            update_archive(candidate, iteration + 1)
        best_score = archive[0]["score"] if archive else None
        print(f"[archive] size={len(archive)} best_score={best_score}")

        # Journal EVERY design this iteration (winners, losers, and exceptions alike) - a record
        # of what was tried and what happened, not just a list of complaints from the losers.
        for candidate in results:
            feedback_history.append(_journal_entry(candidate, iteration + 1))

        iter_best = max(results, key=_candidate_score)
        print(f"\nIteration {iteration + 1} best design: {iter_best['label']} "
              f"(exec={iter_best['exec_pass']}, req={iter_best['req_pass']}, "
              f"req_score={iter_best.get('req_score', 0.0):.2f})")

        if iter_best["req_pass"]:
            print(f"\n{'=' * 80}")
            print(f"[OK] Script is production-ready! (design {iter_best['label']})")
            print_artifacts(iter_best)
            print(f"{'=' * 80}\n")
            return iter_best["script"]

    print(f"\n{'=' * 80}")
    print("[WARNING] Max iterations reached. Returning best effort.")
    if best_candidate:
        status = "executed + requirements" if best_candidate["req_pass"] else (
            f"executed cleanly, req_score={best_candidate.get('req_score', 0.0):.2f}"
            if best_candidate["exec_pass"] else "did not execute")
        print(f"Best candidate: iteration {best_candidate['iteration']}, design {best_candidate['label']} ({status}).")
        print_artifacts(best_candidate)
    print(f"{'=' * 80}\n")
    return best_candidate["script"] if best_candidate else None
