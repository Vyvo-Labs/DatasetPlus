# DatasetPlus

Enhanced dataset management utilities for Hugging Face Hub, with support for audio data processing.

## Installation

```bash
# Install from PyPI
uv pip install datasetplus

# Or install from source
git clone https://github.com/kadirnar/datasetplus.git
cd datasetplus
uv pip install ".[dev]"
```

## Features

- Easy dataset download from Hugging Face Hub
- Audio data processing from Parquet files
- Colored logging system
- Progress tracking

## Quick Start

```python
from datasetplus import HFDatasetManager, AudioProcessor

# Download dataset
dataset = HFDatasetManager()
dataset.download(
    repo_id="fixie-ai/llama-questions",
    local_dir="output/llama_questions",
    repo_type="dataset",
)

# Process audio data
processor = AudioProcessor("path/to/parquet")
metadata = processor.get_metadata()
audio_files = processor.extract_audio_files("output_dir", limit=5)
```

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/kadirnar/datasetplus.git
cd datasetplus
```

2. Install dependencies with Poetry:

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

3. Set up pre-commit hooks:

```bash
# Install pre-commit if you haven't already
pip install pre-commit

# Install the pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

The pre-commit hooks will automatically:

- Format code with Ruff
- Run type checking with mypy
- Check for common issues with pre-commit hooks

## Code Style

This project uses:

- Ruff for code formatting and linting (replacing Black, isort, and flake8)
- mypy for type checking

Ruff is configured to:

- Follow Black-compatible formatting
- Check import sorting
- Enforce docstring standards
- Check for common code issues
- Apply automatic fixes when possible

All of these are automatically run via pre-commit hooks.
