"""Entry point for the multi-agent code-generation pipeline.

Supports multiple domain configs (bioimage, trello, etc.) via --config flag.
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from pipeline import generate_and_optimize


async def main(report_path: str, data_dir: str, output_dir: str, max_iterations: int, designs_per_iteration: int):
    """Run the pipeline on a task report with domain-specific configuration."""
    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()

    final_script = await generate_and_optimize(
        report=report_content,
        config=CONFIG,
        data_dir=data_dir,
        max_iterations=max_iterations,
        output_dir=output_dir,
        designs_per_iteration=designs_per_iteration,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"analysis_script_{timestamp}.py"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_script)

    print("\n" + "=" * 80)
    print("FINAL COMPILED SCRIPT")
    print("=" * 80)
    print(f"\nScript saved to: {output_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-agent code-generation pipeline"
    )
    parser.add_argument(
        "--config",
        default="bioimage",
        choices=["bioimage", "trello"],
        help="Domain configuration to use (default: bioimage)"
    )
    parser.add_argument(
        "--report",
        help="Path to task report file"
    )
    parser.add_argument(
        "--data-dir",
        help="Path to input data directory"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Path to output directory for generated script"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum redesign iterations (default: 2). Total full design attempts run is "
             "max-iterations x designs-per-iteration, each with its own orchestrator + compiler "
             "calls (Opus for both, in the trello config) - this multiplies fast."
    )
    parser.add_argument(
        "--designs-per-iteration",
        type=int,
        default=3,
        help="Independent parallel design attempts fanned out per iteration (default: 3); "
             "the best-scoring one is kept. Set to 1 for the classic single-design-per-iteration behavior."
    )

    args = parser.parse_args()

    # Load config module
    if args.config == "bioimage":
        from bioimage_config import CONFIG
        report_default = "./inputs/report/report_20260710_202254.md"
        data_dir_default = "./inputs/images"
    elif args.config == "trello":
        from trello_config import CONFIG
        report_default = "./inputs/trello_reports/task_report.md"
        data_dir_default = "./inputs/trello_data"
    else:
        raise ValueError(f"Unknown config: {args.config}")

    # Use defaults if not specified
    report_path = args.report or report_default
    data_dir = args.data_dir or data_dir_default

    asyncio.run(main(report_path, data_dir, args.output_dir, args.max_iterations, args.designs_per_iteration))
