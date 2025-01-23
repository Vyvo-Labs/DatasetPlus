from datasetplus import HFDatasetManager


def main() -> None:
    """Download a dataset from Hugging Face Hub."""
    dataset = HFDatasetManager()
    dataset.download(
        repo_id="bookbot/ljspeech_phonemes",
        local_dir="output/ljspeech_phonemes",
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
