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
from datasetplus import DatasetPlus, AudioProcessor

# Download dataset
dataset = DatasetPlus()
dataset.download(
    repo_id="fixie-ai/llama-questions",
    local_dir="output/llama_questions",
    repo_type="dataset"
)

# Process audio data
processor = AudioProcessor("path/to/parquet")
metadata = processor.get_metadata()
audio_files = processor.extract_audio_files("output_dir", limit=None) # Set to None to extract all audio files
```
