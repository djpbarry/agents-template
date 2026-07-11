"""Entry point for the multi-agent code-generation pipeline.

Swap bioimage_config for a different domain config to retarget the pipeline.
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from pipeline import generate_and_optimize
from bioimage_config import CONFIG  # <- Swap this import to use a different domain config


async def main(report_path: str, data_dir: str, output_dir: str, max_iterations: int):
    """Run the pipeline on a task report with domain-specific configuration."""
    with open(report_path, 'r') as f:
        report_content = f.read()

    final_script = await generate_and_optimize(
        report=report_content,
        config=CONFIG,
        data_dir=data_dir,
        max_iterations=max_iterations
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"analysis_script_{timestamp}.py"

    with open(output_file, 'w') as f:
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
        "--report",
        default="./inputs/report/report_20260710_202254.md",
        help="Path to task report file"
    )
    parser.add_argument(
        "--data-dir",
        default="./inputs/images",
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
        default=5,
        help="Maximum refinement iterations"
    )

    args = parser.parse_args()
    asyncio.run(main(args.report, args.data_dir, args.output_dir, args.max_iterations))
