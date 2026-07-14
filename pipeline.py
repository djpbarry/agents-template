"""Generic multi-agent code-generation pipeline.

Orchestrator → Workers (parallel) → Compiler → Evaluator, with feedback loop.
Agnostic to domain — configure via PipelineConfig.
"""

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from config import PipelineConfig

load_dotenv(override=True)
async_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# System prompts for role-based agents (generic, domain-agnostic)
ORCHESTRATOR_SYSTEM = """You are an expert data analysis solutions architect. Your role is to design minimal, modular architectures.
- Prioritize simplicity and clear separation of concerns
- Design only essential functions
- Each function should have a single, well-defined responsibility
- Your designs are the blueprint for implementation"""

WORKER_SYSTEM = """You are an expert Python developer. Your role is to implement functions to specification.
- Write clean, minimal code
- Follow the function specification exactly
- No extra functions, no over-engineering
- Reuse other architecture functions when appropriate
- Each function should be production-ready and independently testable"""

COMPILER_SYSTEM = """You are an expert code integrator. Your role is to assemble modular functions into a cohesive script.
- Consolidate overlapping functions
- Remove redundancy and dead code
- Strip unnecessary complexity
- Ensure all functions work together seamlessly
- The output should be minimal, clean, and production-ready"""

EVALUATOR_SYSTEM = """You are an expert code reviewer and validator. Your role is to verify code meets requirements and works correctly.
- Assess task alignment, code quality, and execution correctness
- Check both the code and its actual behavior (if available)
- Be critical but fair - flag real issues, not style preferences
- Provide actionable feedback for improvement
- Your verdict determines if the code is production-ready"""

# Message prompts for LLM invocations (generic templates with placeholders for domain-specific content)
ORCHESTRATOR_PROMPT = """
You are an experienced data analysis architect. Design a minimal, focused approach for this task.

Report: {report}

Input Data: {input_data}

{feedback}

STEP 1: ANALYZE THE DATA
Examine the available fields and structures.

STEP 2: SUGGEST METRICS
Suggest 5-7 metrics that answer the guiding questions in the report. Focus on:
- Timing metrics (how long things take)
- Distribution metrics (how work is spread)
- Staleness/health metrics (active vs. inactive)
- Any custom data (labels, fields, etc.)

STEP 3: PLAN VISUALIZATIONS
Plan 3+ visualizations for your metrics. Prefer:
- Time-series or trend plots
- Heatmaps for multi-dimensional data
- Scatter plots or distribution plots

Avoid simple bar/pie charts unless essential.

STEP 4: DESIGN MINIMAL ARCHITECTURE
Design only load_data() and main(). main() does all metric computation and visualization.

Return your response in this format:

<analysis>
1. Describe the data structure briefly
2. List your 5-7 metrics and brief rationale for each
3. List your visualizations (names and what they show)
4. Brief overview of how main() will work
</analysis>

<tasks>
    <task>
    <function>main</function>
    <description>Load data, compute metrics, print results, save 3+ PNG visualizations</description>
    <input>None</input>
    <output>None</output>
    </task>
    <task>
    <function>load_data</function>
    <description>Load the Trello JSON export from data directory</description>
    <input>None</input>
    <output>Parsed board data</output>
    </task>
</tasks>
"""

WORKER_PROMPT = """
Implement the {function} function. Be direct—no defensive coding.

Architecture: {description}
Input: {input}
Output: {output}

Task: {original_report}
Data: {input_data}
Libraries: {library_notes}
Domain: {domain_notes}

CRITICAL RULES:
1. Implement ONLY the function '{function}', no helpers
2. Fail fast: if required data is missing, raise an error
3. No try/except unless absolutely necessary
4. One-line docstrings only
5. Clean, simple, direct code
6. Use only listed libraries + standard library
7. If implementing main(): make its FIRST line `sys.stdout.reconfigure(encoding='utf-8')` (and import sys) so UTF-8 output works on all platforms

Wrap your function in <response> tags like this:

<response>
def function_name(args):
    # docstring and code here
</response>

The tags are metadata markers only—do not include them in the actual Python code.
"""

COMPILER_PROMPT = """
Integrate these functions into one complete, executable Python script.

Architecture: {analysis}

Functions:
{functions}

Libraries: {library_notes}
{error_feedback}
RULES:
1. Write complete Python code (imports → functions → main() call)
2. One-line docstrings only
3. Minimal, clean code (no defensive try/except unless critical)
4. Remove duplicate/unused functions

ENCODING (MANDATORY - always include these, non-negotiable):
- Line 1 MUST be exactly: # -*- coding: utf-8 -*-
- After imports, the FIRST line of main() MUST be: sys.stdout.reconfigure(encoding='utf-8')
- Always import sys
- You may freely use UTF-8 characters (—, ✓, →, etc.); the above guarantees they work on all platforms

Wrap the complete script in <response> tags exactly like this:

<response>
# -*- coding: utf-8 -*-
import sys
import ...

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ...

if __name__ == '__main__':
    main()
</response>

The <response> tags are METADATA MARKERS ONLY—do not include them in the Python code itself.
"""

EXECUTION_VALIDATOR_PROMPT = """
Check if this Python script executes without errors.

Script: {content}
Execution Result: {execution_result}

PASS if: Script ran without any errors or exceptions.
FAIL if: Any error occurred (SyntaxError, FileNotFoundError, RuntimeError, etc.).

<evaluation>
PASS or FAIL
</evaluation>

<feedback>
If PASS: "Script executed successfully."
If FAIL: "[Exact error message and which part of the code needs to be fixed]"
</feedback>
"""

REQUIREMENTS_VALIDATOR_PROMPT = """
Check if this successfully-executed script produces the required output.

Task: {report}
Script: {content}
Execution Output: {execution_result}

PASS if ALL are true:
1. Script produced 5+ metrics (visible in the console output above)
2. The "Files actually produced on disk" listing contains 3+ PNG files, each with a non-zero byte size
   (judge by the actual file listing, NOT by what the code claims to write — a 0-byte or missing PNG FAILS)
3. Visualizations have titles and axis labels (from the code)
4. Code is clean (one-line docstrings, no bloat)

FAIL if any criterion is not met.

<evaluation>
PASS or FAIL
</evaluation>

<feedback>
If PASS: "All requirements met. Data gaps for future analysis: [list 2-3 missing fields that would help answer the guiding questions]"
If FAIL: "[Specific requirement not met and what needs to be added]"
</feedback>
"""


# Core LLM interface
async def llm_call(prompt: str, system_prompt: str = None, model: str = None, cache_prompt: bool = False, max_tokens: int = 8192) -> str:
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


def parse_tasks(tasks_xml: str) -> list[dict]:
    """Parse XML tasks into a list of task dictionaries."""
    tasks = []
    try:
        root = ET.fromstring(f"<root>{tasks_xml}</root>")
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
        # Fallback: try to extract tasks manually using regex
        import re
        task_pattern = r'<task>(.*?)</task>'
        for match in re.finditer(task_pattern, tasks_xml, re.DOTALL):
            task_content = match.group(1)
            task = {}
            func_match = re.search(r'<function>(.*?)</function>', task_content)
            desc_match = re.search(r'<description>(.*?)</description>', task_content)
            if func_match:
                task['function'] = func_match.group(1).strip()
            if desc_match:
                task['description'] = desc_match.group(1).strip()
            if task:
                tasks.append(task)
    return tasks


# Sandbox flags for running untrusted, LLM-generated code. Docker here provides both
# dependency pinning AND isolation. Tune these if a host/platform rejects a flag.
DOCKER_SANDBOX_FLAGS = [
    "--network", "none",              # no network access
    "--memory", "1g",                 # cap RAM
    "--memory-swap", "1g",            # == memory, so swap is disabled
    "--cpus", "2",                    # cap CPU
    "--pids-limit", "256",            # limit processes (fork-bomb guard)
    "--read-only",                    # read-only root filesystem
    "--cap-drop", "ALL",              # drop all Linux capabilities
    "--security-opt", "no-new-privileges",  # block privilege escalation
    "--user", "1000:1000",            # run as non-root
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
async def compile_script(orchestrator_results: dict, config: PipelineConfig, error_feedback: str = "") -> str:
    """Compile worker functions into a single executable script, optionally fixing a prior execution error."""
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

    compiler_input = COMPILER_PROMPT.format(
        analysis=analysis,
        functions=functions_text,
        library_notes=config.available_libraries,
        error_feedback=error_section,
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


async def validate_execution(compiled_script: str, config: PipelineConfig, data_dir: str = None,
                             artifacts_dir: str = None) -> tuple[str, str, str, list[dict]]:
    """Check if script executes. Returns (PASS/FAIL, feedback, execution_output, artifacts)."""
    execution_result = "(Docker unavailable)"
    exec_output = ""
    artifacts = []

    if data_dir:
        exec_success, exec_output, artifacts = execute_script_in_docker(
            compiled_script, data_dir, config.docker_image, artifacts_dir=artifacts_dir)
        if exec_success is None:
            execution_result = f"Docker unavailable: {exec_output}"
        elif exec_success:
            # Keep the TAIL: final printed output (incl. data-gap suggestions) matters most
            execution_result = f"Script executed successfully.\n\nOutput:\n{exec_output[-2000:]}"
        else:
            # Keep the TAIL: Python puts the actual exception last, after the traceback frames
            execution_result = f"Script execution failed.\n\nError:\n{exec_output[-2000:]}"

    validator_input = EXECUTION_VALIDATOR_PROMPT.format(
        content=compiled_script,
        execution_result=execution_result
    )

    validator_response = await llm_call(validator_input, system_prompt=EVALUATOR_SYSTEM,
                                       model=config.executor_evaluator_model, cache_prompt=True)
    evaluation = extract_xml(validator_response, "evaluation").strip()
    feedback = extract_xml(validator_response, "feedback").strip()

    if not evaluation:
        print(f"DEBUG: Execution validator response (first 800 chars):\n{validator_response[:800]}")
        # Fallback: look for bare PASS/FAIL if tags are missing
        if re.search(r'\bFAIL\b', validator_response):
            evaluation = "FAIL"
        elif re.search(r'\bPASS\b', validator_response):
            evaluation = "PASS"
        if not feedback:
            feedback = validator_response.strip()

    return evaluation, feedback, exec_output, artifacts


def _format_artifacts(artifacts: list[dict]) -> str:
    """Render the list of produced files with sizes; flag empty files as suspect."""
    if not artifacts:
        return "(No files were produced by the script.)"
    lines = []
    for a in artifacts:
        flag = "  [WARNING: 0 bytes - likely not a valid image]" if a["size"] == 0 else ""
        lines.append(f"- {a['name']} ({a['size']} bytes){flag}")
    return "\n".join(lines)


async def validate_requirements(compiled_script: str, report: str, exec_output: str, config: PipelineConfig,
                                artifacts: list[dict] = None) -> tuple[str, str]:
    """Check if script output meets requirements. Returns (PASS/FAIL, feedback)."""
    artifacts_listing = _format_artifacts(artifacts or [])
    validator_input = REQUIREMENTS_VALIDATOR_PROMPT.format(
        report=report,
        content=compiled_script,
        # Keep the TAIL: the script prints metrics then data-gap suggestions at the very end
        execution_result=f"Console output:\n{exec_output[-3000:]}\n\nFiles actually produced on disk:\n{artifacts_listing}"
    )

    validator_response = await llm_call(validator_input, system_prompt=EVALUATOR_SYSTEM,
                                       model=config.requirements_evaluator_model, cache_prompt=True)
    evaluation = extract_xml(validator_response, "evaluation").strip()
    feedback = extract_xml(validator_response, "feedback").strip()

    if not evaluation:
        print(f"DEBUG: Requirements validator response (first 800 chars):\n{validator_response[:800]}")
        # Fallback: look for bare PASS/FAIL if tags are missing
        if re.search(r'\bFAIL\b', validator_response):
            evaluation = "FAIL"
        elif re.search(r'\bPASS\b', validator_response):
            evaluation = "PASS"
        if not feedback:
            feedback = validator_response.strip()

    return evaluation, feedback


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
    """Rank candidates lexicographically: requirements-pass, then execution-pass, then valid-PNG count.

    Tuples compare left-to-right and bools sort as 0/1, so a requirements-passing script always
    wins; among equal pass/fail status, the one that produced more non-zero-byte PNGs ranks higher.
    """
    valid_pngs = sum(1 for a in candidate["artifacts"]
                     if a["name"].lower().endswith(".png") and a["size"] > 0)
    return (candidate["req_pass"], candidate["exec_pass"], valid_pngs)


async def generate_and_optimize(report: str, config: PipelineConfig, data_dir: str = None,
                                max_iterations: int = 2, output_dir: str = None) -> str:
    """Orchestrate → Compile → Evaluate → Redesign loop until script is production-ready."""
    # Accumulated failure history so each orchestrator redesign sees ALL prior issues,
    # not just the most recent one (prevents fix-A-breaks-B oscillation).
    feedback_history = []
    # Best script seen so far, kept across iterations so a timeout returns the best
    # candidate (not merely whatever the last iteration produced).
    best_candidate = None
    input_metadata = config.extract_input_metadata(data_dir) if data_dir else "(No input data provided)"
    # Base dir under which each iteration gets its OWN artifacts/iter_N subdir, so the
    # files on disk always match the script we ultimately return.
    artifacts_base = str(Path(output_dir) / "artifacts") if output_dir else None

    def record_candidate(script, exec_pass, req_pass, iteration, artifacts_dir, artifacts):
        """Keep the highest-scoring candidate; ties resolve to the earliest (avoids regressions)."""
        nonlocal best_candidate
        candidate = {
            "script": script, "exec_pass": exec_pass, "req_pass": req_pass,
            "iteration": iteration, "artifacts_dir": artifacts_dir, "artifacts": artifacts,
        }
        if best_candidate is None or _candidate_score(candidate) > _candidate_score(best_candidate):
            best_candidate = candidate

    for iteration in range(max_iterations):
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'=' * 80}")

        if feedback_history:
            print(f"\nRedesigning based on feedback...")
            joined = "\n\n".join(feedback_history)
            feedback_section = (
                "Previous attempts had the issues below. Address ALL of them at once; "
                "do NOT reintroduce an earlier problem while fixing a later one:\n"
                f"{joined}"
            )
        else:
            feedback_section = ""

        orchestrator_input = format_prompt(
            ORCHESTRATOR_PROMPT,
            report=report,
            input_data=input_metadata,
            feedback=feedback_section,
        )

        orchestrator_response = await llm_call(orchestrator_input, system_prompt=ORCHESTRATOR_SYSTEM,
                                               model=config.orchestrator_model, cache_prompt=True)
        analysis = extract_xml(orchestrator_response, "analysis")
        tasks_xml = extract_xml(orchestrator_response, "tasks")
        tasks = parse_tasks(tasks_xml)

        print(f"\nArchitecture: {len(tasks)} functions")

        print("Generating worker implementations...")
        worker_results = await asyncio.gather(
            *[_call_worker(task_info, i, report, input_metadata, config) for i, task_info in
              enumerate(tasks, 1)]
        )

        orchestrator_results = {
            "analysis": analysis,
            "worker_results": worker_results,
        }

        # INNER LOOP: Compiler + Execution Validator (up to 3 compilation attempts)
        MAX_COMPILE_ATTEMPTS = 3
        compiled_script = None
        exec_output = None
        artifacts = []
        execution_passed = False
        # Each iteration writes to its own subdir so a returned earlier script's files
        # aren't clobbered by a later iteration's run.
        iter_artifacts_dir = str(Path(artifacts_base) / f"iter_{iteration + 1}") if artifacts_base else None

        compile_error = ""
        for compile_attempt in range(MAX_COMPILE_ATTEMPTS):
            print(f"\n  Compile attempt {compile_attempt + 1}/{MAX_COMPILE_ATTEMPTS}...")
            compiled_script = await compile_script(orchestrator_results, config, error_feedback=compile_error)

            print("  Validating execution...")
            exec_verdict, exec_feedback, exec_output, artifacts = await validate_execution(
                compiled_script, config, data_dir, artifacts_dir=iter_artifacts_dir)
            print(f"  Execution: {exec_verdict}")
            feedback_safe = exec_feedback.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('•', '-')
            print(f"  Feedback: {feedback_safe}")

            if exec_verdict == "PASS":
                execution_passed = True
                break

            # Execution failed - pass the error to the next compile attempt
            if compile_attempt < MAX_COMPILE_ATTEMPTS - 1:
                compile_error = exec_feedback

        if not execution_passed:
            print(f"\n  [FAILED] Could not compile working script after {MAX_COMPILE_ATTEMPTS} attempts.")
            # Still a candidate (a broken script beats nothing), but ranked lowest.
            record_candidate(compiled_script, exec_pass=False, req_pass=False, iteration=iteration + 1,
                             artifacts_dir=iter_artifacts_dir, artifacts=artifacts)
            feedback_history.append(
                f"[Iteration {iteration + 1}] Execution failed after {MAX_COMPILE_ATTEMPTS} "
                f"compile attempts: {exec_feedback}"
            )
            continue

        # STAGE 2: Check requirements (only if execution passed)
        print("\nValidating requirements...")
        req_verdict, req_feedback = await validate_requirements(
            compiled_script, report, exec_output, config, artifacts=artifacts)
        print(f"Requirements: {req_verdict}")
        feedback_safe = req_feedback.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('•', '-')
        print(f"Feedback: {feedback_safe}")

        req_passed = req_verdict == "PASS"
        record_candidate(compiled_script, exec_pass=True, req_pass=req_passed, iteration=iteration + 1,
                         artifacts_dir=iter_artifacts_dir, artifacts=artifacts)

        if req_passed:
            print(f"\n{'=' * 80}")
            print("[OK] Script is production-ready!")
            if artifacts and iter_artifacts_dir:
                print(f"Produced {len(artifacts)} file(s) in: {iter_artifacts_dir}")
                for a in artifacts:
                    print(f"  - {a['name']} ({a['size']} bytes)")
            print(f"{'=' * 80}\n")
            return compiled_script

        # Requirements failed - add to history so the next redesign sees it alongside the rest
        feedback_history.append(
            f"[Iteration {iteration + 1}] Executed cleanly but requirements not met: {req_feedback}"
        )

    print(f"\n{'=' * 80}")
    print("[WARNING] Max iterations reached. Returning best effort.")
    if best_candidate:
        status = "executed + requirements" if best_candidate["req_pass"] else (
            "executed cleanly" if best_candidate["exec_pass"] else "did not execute")
        print(f"Best candidate: iteration {best_candidate['iteration']} ({status}).")
        if best_candidate["artifacts"] and best_candidate["artifacts_dir"]:
            print(f"Produced {len(best_candidate['artifacts'])} file(s) in: {best_candidate['artifacts_dir']}")
            for a in best_candidate["artifacts"]:
                print(f"  - {a['name']} ({a['size']} bytes)")
    print(f"{'=' * 80}\n")
    return best_candidate["script"] if best_candidate else compiled_script


