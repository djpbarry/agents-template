# BIA-TEAM

Multi-agent system that generates, validates, and optimizes Python scripts for bioimage analysis tasks. Given a task report and sample images, it orchestrates Claude agents to design an architecture, implement it, and iteratively refine it until the script passes validation.

## How It Works

```
Task Report + Sample Images
        ↓
Orchestrator (architecture design)
        ↓
Workers (parallel implementation)
        ↓
Compiler (assemble into one script)
        ↓
Evaluator (Docker execution + validation)
        ↓
  Pass → Final Script
  Fail → Feedback loop back to Orchestrator
```

- **Models**: Opus 4.8 (architecture), Sonnet 5 (compilation/evaluation), Haiku 4.5 (implementation) — matched to task complexity
- **Execution**: scripts run in a pre-built Docker image (`bia-analysis:latest`) with dependencies pinned, so generated code can only use pre-installed libraries (numpy, scipy, scikit-image, scikit-learn, pandas, bioio, bioio-tifffile)
- **Structured I/O**: XML-tagged prompts/responses for reliable parsing

## Setup

Requirements: Python 3.11+, Docker Desktop running, an Anthropic API key.

```bash
pip install -r requirements.txt
docker build -t bia-analysis:latest .
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

Place a task report and sample TIFFs under `inputs/`:

```
inputs/report/report_YYYYMMDD_HHMMSS.md
inputs/images/*.tif
```

Then run:

```bash
python app.py
```

The final validated script is written to `outputs/analysis_script_<timestamp>.py`.

## Limitations

- Generated scripts are restricted to the pre-installed library set (no external packages)
- Execution timeout: 300s; max 5 redesign iterations (both configurable in `app.py`)
- Docker is required to validate execution — without it, evaluation skips the run step and checks code quality only

## License

MIT
