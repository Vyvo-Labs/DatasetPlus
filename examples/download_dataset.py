"""Example script for downloading a dataset from Hugging Face Hub."""

from datasetplus import HFDatasetManager


def main() -> None:
    """Download a dataset from Hugging Face Hub."""
    dataset = HFDatasetManager()
    dataset.download(
        repo_id="fixie-ai/llama-questions",
        local_dir="output/llama_questions",
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
