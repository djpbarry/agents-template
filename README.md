# BIA-TEAM

An AI-powered system for generating, validating, and optimizing Python scripts for bioimage analysis tasks. Uses multi-agent orchestration and Docker-based execution to iteratively refine analysis pipelines.

## Overview

BIA-TEAM automates the creation of image analysis scripts by:
1. **Parsing** a task report to understand analysis requirements
2. **Extracting** metadata from sample TIFF images (channels, dimensions, data types)
3. **Orchestrating** a team of Claude LLM agents to design and implement the analysis pipeline
4. **Validating** generated scripts via Docker execution
5. **Optimizing** through iterative feedback until the script meets requirements

## Features

- **Multi-Agent Orchestration**: Specialized agents for architecture design, implementation, code compilation, and validation
- **Docker-Based Execution**: Scripts run in an isolated, pre-configured Docker environment with all dependencies pre-installed
- **Iterative Optimization**: Evaluator feedback loops back to the orchestrator for automatic architecture redesign
- **Bioio Integration**: Reads multi-channel TIFF images with full metadata extraction
- **Minimal Code Generation**: Produces clean, focused scripts with no over-engineering or visualization code
- **Structured Prompting**: Uses XML-based structured output for reliable parsing of LLM responses

## How It Works

```
Task Report + Sample Images
          ↓
   Extract Image Metadata
          ↓
Orchestrator (Architecture Design)
          ↓
    Workers (Implementation)
          ↓
   Compiler (Code Assembly)
          ↓
   Evaluator (Docker Execution + Validation)
          ↓
    [Passes?] → Final Script
          ↓
    [Fails] → Feedback → Redesign
```

## Technical Stack

- **LLMs**: Claude Opus 4.8 (orchestration), Claude Sonnet 5 (compilation/evaluation), Claude Haiku 4.5 (implementation)
- **Image Processing**: bioio, bioio-tifffile, scikit-image, numpy, scipy
- **Execution**: Docker (containerized, dependency-stable environment)
- **Code Integration**: Python 3.13, standard library
- **Data Handling**: pandas, scikit-learn for analysis results

## Setup

### Requirements

- Python 3.11+
- Docker Desktop (running)
- BIOS virtualization enabled (Windows)
- Anthropic API key

### Installation

```bash
# Clone repository
git clone <repo-url>
cd bia_team

# Install dependencies
pip install -r requirements.txt

# Configure environment
export ANTHROPIC_API_KEY="your-key-here"
```

### Docker Image

Pre-built Docker image with all analysis libraries:

```bash
docker build -t bia-analysis:latest .
```

Image includes: numpy, scipy, scikit-image, scikit-learn, pandas, bioio, bioio-tifffile

## Usage

### Input Structure

```
inputs/
  report/
    report_YYYYMMDD_HHMMSS.md  # Task requirements
  images/
    *.tif                       # Sample TIFF images
```

### Running

```python
from app import generate_and_optimize, extract_image_metadata

report = open('./inputs/report/report.md').read()
image_metadata = extract_image_metadata('./inputs/images')
script = generate_and_optimize(
    report=report,
    image_metadata=image_metadata,
    image_dir='./inputs/images',
    max_iterations=5
)

# Generated script is saved to outputs/
```

### Output

- Final validated Python script
- Execution logs and evaluation feedback
- Architecture analysis and design decisions

## Architecture

**FlexibleOrchestrator**: Coordinates the multi-agent pipeline
- Calls Orchestrator to design minimal architecture
- Spawns Workers to implement each function in parallel
- Calls Compiler to assemble functions into a single script
- Calls Evaluator to validate against requirements

**Key Design Decisions**

- Static library constraints (no dynamic pip installs)
- Pre-baked Docker image (faster execution, consistent environment)
- Feedback-driven redesign (Evaluator → Orchestrator)
- Role-specific model selection (complexity-appropriate)

## Constraints & Limitations

- Scripts restricted to pre-installed libraries (no external packages)
- Execution timeout: 300 seconds
- Maximum 5 redesign iterations (configurable)
- Requires Docker to validate script execution

## License

MIT
