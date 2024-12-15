from datasetplus import DatasetPlus

# Initialize DatasetPlus (if you have a token, pass it here)
dataset = DatasetPlus()

# Download the dataset to a local directory
dataset.download(
    repo_id="fixie-ai/llama-questions",
    local_dir="./output/llama_questions",
    repo_type="dataset",  # Explicitly specify that we're downloading a dataset
    ignore_patterns=[".gitignore", "README.md", ".gitattributes"]  # Ignore specific files and folders
)

