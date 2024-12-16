# DatasetPlus

Enhanced dataset management utilities for the Hugging Face Hub.

## Installation

```bash
uv pip install -e .
```

## Usage Examples

### Download a Dataset

```python
from datasetplus import Dataset

dataset = Dataset()
dataset.download(
    repo_id="fixie-ai/llama-questions",
    local_dir="output/llama_questions",
    repo_type="dataset",
)
```

### Process Audio Data

```python
from datasetplus import AudioProcessor

processor = AudioProcessor("path/to/audio/files")
processor.process()
```
