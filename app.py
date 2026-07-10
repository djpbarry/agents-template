# Reproduced from https://github.com/anthropics/claude-cookbooks/blob/main/patterns/agents/util.py

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from bioio import BioImage
from dotenv import load_dotenv

load_dotenv(override=True)
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
DEFAULT_MODEL = "claude-haiku-4-5"


def llm_call(prompt: str, system_prompt: str = "", model=DEFAULT_MODEL) -> str:
    """
    Calls the model with the given prompt and returns the response.

    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str, optional): The system prompt to send to the model. Defaults to "".
        model (str, optional): The model to use for the call.

    Returns:
        str: The response from the language model.
    """
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        temperature=0.1,
    )
    return response.content[0].text


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


def extract_image_metadata(directory: str) -> str:
    """Extract metadata from multi-channel TIFF images using bioio."""
    metadata_list = []

    for file_path in sorted(Path(directory).glob('*.tif*')):
        try:
            img = BioImage(str(file_path))
            metadata = {
                "filename": file_path.name,
                "shape": dict(zip(img.dims.order, img.shape)),
                "dtype": str(img.dtype),
                "ndim": len(img.shape),
            }

            if hasattr(img, 'channel_names') and img.channel_names:
                metadata["channel_names"] = list(img.channel_names)

            metadata_list.append(metadata)
        except Exception as e:
            metadata_list.append({
                "filename": file_path.name,
                "error": str(e)
            })

    return str(metadata_list)


def parse_tasks(tasks_xml: str) -> list[dict]:
    """Parse XML tasks into a list of task dictionaries."""
    tasks = []
    try:
        root = ET.fromstring(f"<root>{tasks_xml}</root>")
        for task_elem in root.findall("task"):
            task = {}
            for child in task_elem:
                if child.text:
                    task[child.tag] = child.text.strip()
            if task:
                tasks.append(task)
    except ET.ParseError:
        pass
    return tasks


EVALUATOR_PROMPT = """
Evaluate if this Python script meets the task requirements and quality standards.

Original Task Report:
{report}

Script to evaluate:
{content}

PASS if ALL of the following are true:
1. TASK ALIGNMENT: Script actually addresses the requirements from the report
2. CLEAN ARCHITECTURE: Only core functions present, no unnecessary utilities
3. MINIMAL: No matplotlib, visualization, or image saving code
4. NO OVER-ENGINEERING: Simple algorithms appropriate to task complexity (per report)
5. FOCUSED: No unused metric collection or redundant logic
6. DOCUMENTED: One-line docstrings only
7. SIZED: Number of lines of code is minimal
8. BEHAVIOR: Returns results, doesn't handle I/O (except main)

Return your response in this format:

<evaluation>
PASS or FAIL
</evaluation>

<feedback>
If FAIL, list specific issues to fix (prioritize task alignment first, then code quality).
If PASS, write "Ready for production."
</feedback>
"""


def compile_script(orchestrator_results: dict, model: str = DEFAULT_MODEL) -> str:
    """Compile worker functions into a single executable script."""
    analysis = orchestrator_results["analysis"]

    functions_text = "\n\n".join([
        f"# Function: {result['function']}\n# Description: {result['description']}\n{result['result']}"
        for result in orchestrator_results["worker_results"]
    ])

    compiler_input = COMPILER_PROMPT.format(
        analysis=analysis,
        functions=functions_text,
    )

    compiled_response = llm_call(compiler_input, model=model)
    compiled_script = extract_xml(compiled_response, "response")

    return compiled_script


def evaluate_script(compiled_script: str, report: str, model: str = DEFAULT_MODEL) -> tuple[str, str]:
    """Evaluate if compiled script meets task requirements and quality standards. Returns (verdict, feedback)."""
    evaluator_input = EVALUATOR_PROMPT.format(report=report, content=compiled_script)
    evaluator_response = llm_call(evaluator_input, model=model)
    evaluation = extract_xml(evaluator_response, "evaluation").strip()
    feedback = extract_xml(evaluator_response, "feedback").strip()
    return evaluation, feedback


def generate_and_optimize(report: str, image_metadata: str, model: str = DEFAULT_MODEL, max_iterations: int = 2) -> str:
    """Orchestrate → Compile → Evaluate → Redesign loop until script is production-ready."""
    orchestrator = FlexibleOrchestrator(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,
        worker_prompt=WORKER_PROMPT,
    )

    feedback_context = ""

    for iteration in range(max_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'='*80}")

        # Step 1: Orchestrator designs architecture (with feedback if redesigning)
        if feedback_context:
            print(f"\nRedesigning based on feedback...")
            modified_orchestrator_prompt = ORCHESTRATOR_PROMPT + f"\n\nPrevious design feedback to address:\n{feedback_context}"
            orchestrator_input = orchestrator._format_prompt(
                modified_orchestrator_prompt,
                report=report,
                input_data=image_metadata
            )
        else:
            orchestrator_input = orchestrator._format_prompt(
                ORCHESTRATOR_PROMPT,
                report=report,
                input_data=image_metadata
            )

        orchestrator_response = llm_call(orchestrator_input, model=model)
        analysis = extract_xml(orchestrator_response, "analysis")
        tasks_xml = extract_xml(orchestrator_response, "tasks")
        tasks = parse_tasks(tasks_xml)

        print(f"\nArchitecture: {len(tasks)} functions")

        # Step 2: Workers implement
        print("\nGenerating worker implementations...")
        worker_results = []
        for i, task_info in enumerate(tasks, 1):
            func_name = task_info.get("function", f"task_{i}")
            worker_input = orchestrator._format_prompt(
                WORKER_PROMPT,
                original_report=report,
                function=func_name,
                description=task_info.get("description", ""),
                input=task_info.get("input", ""),
                output=task_info.get("output", ""),
                input_data=image_metadata,
            )
            worker_response = llm_call(worker_input, model=model)
            worker_content = extract_xml(worker_response, "response")
            worker_results.append({
                "function": func_name,
                "description": task_info.get("description", ""),
                "result": worker_content,
            })

        orchestrator_results = {
            "analysis": analysis,
            "worker_results": worker_results,
        }

        # Step 3: Compiler assembles
        print("Compiling script...")
        compiled_script = compile_script(orchestrator_results, model=model)

        # Step 4: Evaluate
        print("Evaluating script...")
        evaluation, feedback = evaluate_script(compiled_script, report=report, model=model)

        print(f"\nEvaluation: {evaluation}")
        print(f"Feedback: {feedback}")

        if evaluation == "PASS":
            print(f"\n{'='*80}")
            print("✓ Script is production-ready!")
            print(f"{'='*80}\n")
            return compiled_script

        feedback_context = feedback

    print(f"\n{'='*80}")
    print("⚠ Max iterations reached. Returning best effort.")
    print(f"{'='*80}\n")
    return compiled_script


class FlexibleOrchestrator:
    """Break down tasks and run them in parallel using worker LLMs."""

    def __init__(
            self,
            orchestrator_prompt: str,
            worker_prompt: str,
            model: str = DEFAULT_MODEL,
    ):
        """Initialize with prompt templates and model selection."""
        self.orchestrator_prompt = orchestrator_prompt
        self.worker_prompt = worker_prompt
        self.model = model

    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with variables."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {e}") from e

    def process(self, report: str, input_data: str) -> dict:
        """Process task by breaking it down and running subtasks in parallel."""

        # Step 1: Get orchestrator response
        orchestrator_input = self._format_prompt(self.orchestrator_prompt, report=report, input_data=input_data)
        orchestrator_response = llm_call(orchestrator_input, model=self.model)

        # Parse orchestrator response
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
            print(f"\n{i}. {task_info['function']}")
            print(f"   {task_info['description']}")
            print(f"   {task_info['input']}")
            print(f"   {task_info['output']}")

        print("\n" + "=" * 80)
        print("GENERATING CONTENT")
        print("=" * 80 + "\n")

        # Step 2: Process each task
        worker_results = []
        for i, task_info in enumerate(tasks, 1):
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
            )

            worker_response = llm_call(worker_input, model=self.model)
            worker_content = extract_xml(worker_response, "response")

            # Validate worker response - handle empty outputs
            if not worker_content or not worker_content.strip():
                print(f"⚠️  Warning: Worker '{task_info['type']}' returned no content")
                worker_content = f"[Error: Worker '{task_info['type']}' failed to generate content]"

            worker_results.append(
                {
                    "function": func_name,
                    "description": task_info.get("description", ""),
                    "result": worker_content,
                }
            )

        # Display results
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


ORCHESTRATOR_PROMPT = """
You are an experienced senior software engineer and architect. Design a MINIMAL approach for this image analysis task.

Report: {report}

Image Data (bioio metadata): {input_data}

Do not write any code, only design an approach. Break it into distinct, self-contained, modular sub-tasks.
Each sub-task should specify a function that a colleague will implement. Keep the number of sub-tasks minimal to stay 
within resource constraints.

Design ONLY the essential functions needed. Do NOT design:
- Visualization or plotting functions
- Preprocessing functions separate from core logic
- Metric collection that isn't used in the final output
- Data saving/export functions (the main function returns data)

Return your response in this format:

<analysis>
Explain your understanding of the report and the rationale behind your approach.
Outline clearly how each sub-task contributes to the overall goal.
</analysis>

<tasks>
    <task>
    <function>main</function>
    <description>The main function for analysing the input data using python</description>
    <input>The input parameters required by the main function, if any</input>
    <output>The output returned by the main function, if any</output>
    </task>
    <task>
    <function>load_images</function>
    <description>A function for loading TIF images</description>
    <input>The input parameters required by the load_images function, if any</input>
    <output>The output returned by the load_images function, if any</output>
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

Image Data: {input_data}

CRITICAL CONSTRAINTS:
- Implement ONLY the specified function named '{function}' - no additional functions or helpers
- NO visualization, plotting, image saving, or output files (return data only, let caller handle I/O)
- NO metric collection that isn't used in the function output
- Use bioio.BioImage for image loading where necessary (from bioio import BioImage)
- Reuse other architecture functions when needed
- For main(), call other designed functions rather than reimplementing them
- Keep algorithm choices simple and justified by the report

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

CRITICAL OPTIMIZATION RULES - APPLY STRICTLY:
1. STRIP VISUALIZATION: Remove ALL matplotlib, visualization, image saving code, and utility functions for I/O
2. STRIP UNUSED CODE: Remove functions that don't appear in the architecture
3. DEDUPLICATE: Merge overlapping functions (e.g., load_image + load_images, filter_and_count logic)
4. DOCSTRINGS: One-line summary + Args/Returns only (no Raises, Notes, Examples, lengthy descriptions)
5. NO OVER-ENGINEERING: Minimal error handling, no redundant re-labeling, no unused parameter handling

Create a complete, minimal Python script:
1. Imports only necessary libraries
2. Core functions from architecture only
3. Simple, clear code with justified algorithms
4. Include code to execute main() at the bottom

Target: Clean, complete, production-quality code - nothing more.

Return your response in this format - it MUST include both the opening and closing xml tags:

<response>

# Your complete, executable Python script here

</response>
"""

with open('./inputs/report/report_20260706_182925.md', 'r') as f:
    report_content = f.read()

image_metadata = extract_image_metadata('./inputs/images')

final_script = generate_and_optimize(
    report=report_content,
    image_metadata=image_metadata,
    max_iterations=2
)

output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"analysis_script_{timestamp}.py"

with open(output_file, 'w') as f:
    f.write(final_script)

print("\n" + "=" * 80)
print("FINAL COMPILED SCRIPT")
print("=" * 80)
print(f"\nScript saved to: {output_file}\n")
print(final_script)
