"""Generic multi-agent code-generation pipeline.

Orchestrator → Workers (parallel) → Compiler → Evaluator, with feedback loop.
Agnostic to domain — configure via PipelineConfig.
"""

import asyncio
import os
import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape
import subprocess
import tempfile
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from config import PipelineConfig

load_dotenv(override=True)
async_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# System prompts for role-based agents (generic, domain-agnostic)
ORCHESTRATOR_SYSTEM = """You are an expert software architect. Your role is to design minimal, modular architectures.
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
You are an experienced senior software engineer and architect. Examine the input data structure and design a MINIMAL approach for this task.

Report: {report}

Input Data Available:
{input_data}

{feedback}

STEP 1: ANALYZE THE DATA
First, carefully examine the input_data metadata above to understand what fields and structures are available.
Identify 5-7 meaningful statistics that can be computed from the available data. Examples:
- Counts (total items, items per category/member/status)
- Aggregations (average, sums, distributions)
- Time-based metrics (if timestamps exist: time between events, duration in state)
- Cross-tabulations (breakdowns by multiple dimensions)

STEP 2: PLAN VISUALIZATIONS
For the statistics you identified, plan 3 visualizations that would illuminate the analysis. Examples:
- Bar charts for categorical distributions
- Pie charts for composition
- Time series for trends
- Heatmaps for cross-tabulations

STEP 3: DESIGN THE ARCHITECTURE
Break the task into distinct, self-contained, modular sub-tasks. Each sub-task should specify a function that a colleague will implement.
Keep the number of sub-tasks minimal to stay within resource constraints.

Design ONLY the essential functions needed. Do NOT design:
- Visualization or plotting functions (main() will handle these)
- Preprocessing functions separate from core logic
- Metric collection that isn't used in the final output

Return your response in this format:

<analysis>
1. Describe the data structure and available fields
2. List the 5-7 statistics you suggest computing (with brief rationale for each)
3. List the 3+ visualizations you plan to create and what they show
4. Explain your architectural approach and how each sub-task contributes to the overall goal
</analysis>

<tasks>
    <task>
    <function>main</function>
    <description>The main function for analysing the input data: load it, compute suggested statistics, generate visualizations, and print results</description>
    <input>The input parameters required by the main function, if any</input>
    <output>The output returned by the main function, if any</output>
    </task>
    <task>
    <function>load_data</function>
    <description>A function for loading and parsing the input data</description>
    <input>The input parameters required by the load_data function, if any</input>
    <output>The output returned by the load_data function, if any</output>
    </task>
</tasks>
"""

WORKER_PROMPT = """
Implement a python function based on the architecture design:

Report: {original_report}

Function Name: {function}
Description: {description}
Input: {input}
Output: {output}

Input Data: {input_data}

{library_notes}

{domain_notes}

CRITICAL CONSTRAINTS:
- Implement ONLY the specified function named '{function}' - no additional functions or helpers
- If the function is main(), it may include visualization, plotting, and file I/O if required by the task. Otherwise, avoid I/O and let caller handle it.
- NO metric collection that isn't used in the function output
- Reuse other architecture functions when needed
- For main(), call other designed functions rather than reimplementing them
- Keep algorithm choices simple and justified by the report
- ONLY use pre-installed libraries listed above and standard library
- If you use non-ASCII characters (like bullet points • in strings), add "# -*- coding: utf-8 -*-" at the top of your response

Write minimal code with one-line docstrings. Output a single function with necessary imports only.

Return your response in this format - it MUST include both the opening and closing xml tags:

<response>

# Your complete, executable Python script here

</response>
"""

COMPILER_PROMPT = """
You are an expert Python developer. Integrate these functions into a single, minimal, executable Python script.

Architecture Analysis:
{analysis}

Functions to integrate:
{functions}

{library_notes}

CRITICAL OPTIMIZATION RULES - APPLY STRICTLY:
1. PRESERVE OUTPUT CODE: Keep visualization, plotting, and data saving code if the task requires it (check the report). Only strip if they're unused or contradict the task.
2. STRIP UNUSED CODE: Remove functions that don't appear in the architecture
3. DEDUPLICATE: Merge overlapping functions
4. DOCSTRINGS: One-line summary only, no Args/Returns/Raises/Notes/Examples
5. NO OVER-ENGINEERING: Minimal error handling, no redundant re-labeling, no unused parameter handling
6. ENCODING: If the script contains non-ASCII characters in strings, add "# -*- coding: utf-8 -*-" at the very top of the file

Create a complete, minimal Python script:
1. Imports only necessary libraries (only those listed above and standard library)
2. Core functions from architecture only
3. Simple, clear code with justified algorithms
4. Include code to execute main() at the bottom

Target: Clean, complete, production-quality code - nothing more.

Return your response in this format - it MUST include both the opening and closing xml tags:

<response>

# Your complete, executable Python script here

</response>
"""

EVALUATOR_PROMPT = """
Evaluate if this Python script meets the task requirements and quality standards.

Original Task Report:
{report}

Script to evaluate:
{content}

Execution Result:
{execution_result}

PASS if ALL of the following are true:
1. EXECUTION: Script ran successfully without errors (if Docker was available)
2. TASK ALIGNMENT: Script actually addresses the requirements from the report
3. OUTPUT VALIDITY: Execution produced expected output (statistics printed, visualizations created)
4. CLEAN ARCHITECTURE: Only core functions present, no unnecessary utilities
5. APPROPRIATE TOOLS: Uses visualization/I/O code only if required by the task (no unnecessary extras, but don't omit if the task requires it)
6. NO OVER-ENGINEERING: Simple algorithms appropriate to task complexity (per report)
7. FOCUSED: No unused metric collection or redundant logic
8. DOCUMENTED: One-line docstrings only
9. SIZED: Number of lines of code is minimal and appropriate for the task scope
10. BEHAVIOR: Returns/prints results appropriately for the task context (I/O may be required if the task asks for visualizations or reports)

IF PASS, ALSO IDENTIFY DATA GAPS:
After validating the script passes all criteria, examine what fields or data points were NOT available in the input data but could improve future analysis. Suggest 2-3 specific data gaps with brief explanations of why they would be useful (e.g., "priority levels", "time-to-completion", "effort estimates", "dependencies between tasks", etc.).

Return your response in this format:

<evaluation>
PASS or FAIL
</evaluation>

<feedback>
If FAIL, list specific issues to fix (prioritize execution/output validity first, then task alignment, then code quality).
If PASS, write "Ready for production. Data gaps: [list 2-3 suggestions with brief rationale]"
</feedback>
"""


# Core LLM interface
async def llm_call(prompt: str, system_prompt: str = None, model: str = None, cache_prompt: bool = False) -> str:
    """
    Calls the model with the given prompt and returns the response.

    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str, optional): The system prompt.
        model (str, optional): The model to use for the call.
        cache_prompt (bool): Enable prompt caching for this call.

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
    response = await async_client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_content,
        messages=messages,
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError("No text content in response")


# Helper functions for data extraction and processing
def extract_xml(text: str, tag: str) -> str:
    """Extracts the content of the specified XML tag from the given text."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


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
    return tasks


def execute_script_in_docker(script: str, data_dir: str, docker_image: str, timeout: int = 300) -> tuple[bool, str]:
    """
    Execute script in Docker container to verify it works.
    Returns (success, output_or_error) or (None, message) if Docker unavailable.
    """
    try:
        subprocess.run(["docker", "ps"], capture_output=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, "Docker not available - skipping execution test"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "script.py"
            script_path.write_text(script)

            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{Path(data_dir).absolute()}:/data:ro",
                "-v", f"{tmpdir}:/work",
                "-w", "/work",
                "-e", "INPUT_FOLDER=/data",
                docker_image,
                "python", "script.py"
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=timeout + 30,
                text=True
            )

            if result.returncode == 0:
                return True, result.stdout or "Script executed successfully"
            else:
                return False, result.stderr or "Script execution failed with no error output"

    except subprocess.TimeoutExpired:
        return False, f"Script execution timed out (>{timeout}s)"
    except Exception as e:
        if "daemon" in str(e).lower() or "pipe" in str(e).lower():
            return None, "Docker daemon not running - skipping execution test"
        return False, f"Execution error: {str(e)}"


# Core async functions for the compilation pipeline
async def compile_script(orchestrator_results: dict, config: PipelineConfig) -> str:
    """Compile worker functions into a single executable script."""
    analysis = orchestrator_results["analysis"]

    functions_text = "\n\n".join([
        f"# Function: {result['function']}\n# Description: {result['description']}\n{result['result']}"
        for result in orchestrator_results["worker_results"]
    ])

    compiler_input = COMPILER_PROMPT.format(
        analysis=analysis,
        functions=functions_text,
        library_notes=config.available_libraries,
    )

    compiled_response = await llm_call(compiler_input, system_prompt=COMPILER_SYSTEM, model=config.compiler_model, cache_prompt=True)
    compiled_script = extract_xml(compiled_response, "response")

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


async def evaluate_script(compiled_script: str, report: str, config: PipelineConfig, data_dir: str = None) -> tuple[str, str]:
    """Evaluate if compiled script meets task requirements and quality standards. Returns (verdict, feedback)."""

    execution_status = None
    execution_result = "(Docker unavailable - execution not tested)"

    if data_dir:
        exec_success, exec_output = execute_script_in_docker(compiled_script, data_dir, config.docker_image)

        if exec_success is None:
            execution_result = f"Docker unavailable: {exec_output}"
        elif exec_success:
            execution_result = f"✓ Script executed successfully\n\nOutput:\n{exec_output[:500]}"
        else:
            execution_result = f"✗ Script execution failed\n\nError:\n{exec_output[:500]}"
            execution_status = "FAIL"

    evaluator_input = EVALUATOR_PROMPT.format(
        report=report,
        content=compiled_script,
        execution_result=execution_result
    )

    evaluator_response = await llm_call(evaluator_input, system_prompt=EVALUATOR_SYSTEM, model=config.evaluator_model, cache_prompt=True)
    evaluation = extract_xml(evaluator_response, "evaluation").strip()
    feedback = extract_xml(evaluator_response, "feedback").strip()

    if execution_status == "FAIL":
        evaluation = "FAIL"
        feedback = f"Script execution failed. Error: {execution_result}\n\n{feedback}"

    return evaluation, feedback


async def _call_worker(task_info: dict, task_index: int, report: str, input_metadata: str, config: PipelineConfig, orchestrator: 'FlexibleOrchestrator') -> dict:
    """Call worker for a single task. Used for parallel execution."""
    func_name = task_info.get("function", f"task_{task_index}")
    worker_input = orchestrator._format_prompt(
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
    worker_response = await llm_call(worker_input, system_prompt=WORKER_SYSTEM, model=config.worker_model, cache_prompt=True)
    worker_content = extract_xml(worker_response, "response")
    return {
        "function": func_name,
        "description": task_info.get("description", ""),
        "result": worker_content,
    }


async def generate_and_optimize(report: str, config: PipelineConfig, data_dir: str = None, max_iterations: int = 2) -> str:
    """Orchestrate → Compile → Evaluate → Redesign loop until script is production-ready."""
    orchestrator = FlexibleOrchestrator(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,
        worker_prompt=WORKER_PROMPT,
        config=config,
    )

    feedback_context = ""
    input_metadata = config.extract_input_metadata(data_dir) if data_dir else "(No input data provided)"

    for iteration in range(max_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'='*80}")

        if feedback_context:
            print(f"\nRedesigning based on feedback...")
            feedback_section = f"Previous design feedback to address:\n{feedback_context}"
        else:
            feedback_section = ""

        orchestrator_input = orchestrator._format_prompt(
            ORCHESTRATOR_PROMPT,
            report=report,
            input_data=input_metadata,
            feedback=feedback_section,
            )

        orchestrator_response = await llm_call(orchestrator_input, system_prompt=ORCHESTRATOR_SYSTEM, model=config.orchestrator_model, cache_prompt=True)
        analysis = extract_xml(orchestrator_response, "analysis")
        tasks_xml = extract_xml(orchestrator_response, "tasks")
        tasks = parse_tasks(tasks_xml)

        print(f"\nArchitecture: {len(tasks)} functions")

        print("\nGenerating worker implementations...")
        worker_results = await asyncio.gather(
            *[_call_worker(task_info, i, report, input_metadata, config, orchestrator) for i, task_info in enumerate(tasks, 1)]
        )

        orchestrator_results = {
            "analysis": analysis,
            "worker_results": worker_results,
        }

        print("Compiling script...")
        compiled_script = await compile_script(orchestrator_results, config)

        print("Evaluating script...")
        evaluation, feedback = await evaluate_script(compiled_script, report=report, config=config, data_dir=data_dir)

        print(f"\nEvaluation: {evaluation}")
        # Replace Unicode characters in feedback for compatibility with Windows console
        feedback_safe = feedback.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('•', '-')
        print(f"Feedback: {feedback_safe}")

        if evaluation == "PASS":
            print(f"\n{'='*80}")
            print("[OK] Script is production-ready!")
            print(f"{'='*80}\n")
            return compiled_script

        feedback_context = feedback

    print(f"\n{'='*80}")
    print("[WARNING] Max iterations reached. Returning best effort.")
    print(f"{'='*80}\n")
    return compiled_script


class FlexibleOrchestrator:
    """Break down tasks and run them in parallel using worker LLMs."""

    def __init__(
            self,
            orchestrator_prompt: str,
            worker_prompt: str,
            config: PipelineConfig,
    ):
        """Initialize with prompt templates and config."""
        self.orchestrator_prompt = orchestrator_prompt
        self.worker_prompt = worker_prompt
        self.config = config

    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with variables."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {e}") from e

    async def process(self, report: str, input_data: str) -> dict:
        """Process task by breaking it down and running subtasks in parallel."""

        orchestrator_input = self._format_prompt(
            self.orchestrator_prompt,
            report=report,
            input_data=input_data,
            feedback="",
        )
        orchestrator_response = await llm_call(orchestrator_input, system_prompt=ORCHESTRATOR_SYSTEM, model=self.config.orchestrator_model)

        analysis = extract_xml(orchestrator_response, "analysis")
        tasks_xml = extract_xml(orchestrator_response, "tasks")
        tasks = parse_tasks(tasks_xml)

        print("\n" + "=" * 80)
        print("SOFTWARE ARCHITECT ANALYSIS")
        print("=" * 80)
        print(f"\n{analysis}\n")

        print("\n" + "=" * 80)
        print(f"IDENTIFIED {len(tasks)} SUB-TASKS")
        print("=" * 80)
        for i, task_info in enumerate(tasks, 1):
            print(f"\n{i}. {task_info.get('function', 'unknown')}")
            print(f"   {task_info.get('description', '')}")
            print(f"   {task_info.get('input', '')}")
            print(f"   {task_info.get('output', '')}")

        print("\n" + "=" * 80)
        print("GENERATING CONTENT")
        print("=" * 80 + "\n")

        async def _process_task(i, task_info):
            func_name = task_info.get("function", f"task_{i}")
            print(f"[{i}/{len(tasks)}] Processing: {func_name}...")

            worker_input = self._format_prompt(
                self.worker_prompt,
                original_report=report,
                function=func_name,
                description=task_info.get("description", ""),
                input=task_info.get("input", ""),
                output=task_info.get("output", ""),
                input_data=input_data,
                library_notes=self.config.available_libraries,
                domain_notes=self.config.domain_notes,
            )

            worker_response = await llm_call(worker_input, system_prompt=WORKER_SYSTEM, model=self.config.worker_model)
            worker_content = extract_xml(worker_response, "response")

            if not worker_content or not worker_content.strip():
                print(f"[WARNING] Worker '{func_name}' returned no content")
                worker_content = f"[Error: Worker '{func_name}' failed to generate content]"

            return {
                "function": func_name,
                "description": task_info.get("description", ""),
                "result": worker_content,
            }

        worker_results = await asyncio.gather(
            *[_process_task(i, task_info) for i, task_info in enumerate(tasks, 1)]
        )

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        for i, result in enumerate(worker_results, 1):
            print(f"\n{'-' * 80}")
            print(f"Function {i}: {result['function'].upper()}")
            print(f"{'-' * 80}")
            print(f"\n{result['result']}\n")

        return {
            "analysis": analysis,
            "worker_results": worker_results,
        }
