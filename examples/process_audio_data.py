"""Example script for processing audio data from a Hugging Face dataset."""

from pathlib import Path
from typing import Optional

from datasetplus import AudioProcessor, HFDatasetManager
from datasetplus.utils import get_logger

logger = get_logger(__name__)


def process_audio_data(
    dataset_name: str = "fixie-ai/llama-questions",
    output_dir: str = "output/llama_questions",
    limit: Optional[int] = 5,
) -> None:
    """Process audio data from a Hugging Face dataset.

    Args:
        dataset_name: Name of the Hugging Face dataset.
        output_dir: Directory to store the output.
        limit: Maximum number of files to process. Defaults to 5.
    """
    try:
        # Initialize and download dataset
        dataset = HFDatasetManager()
        dataset.download(
            repo_id=dataset_name,
            local_dir=output_dir,
            repo_type="dataset",
        )

        # Process audio data
        parquet_path = Path(output_dir) / "train.parquet"
        processor = AudioProcessor(parquet_path)

        # Get metadata
        metadata = processor.get_metadata()
        logger.info(f"Dataset metadata: {metadata}")

        # Extract audio files
        audio_files = processor.extract_audio_files(
            output_dir=str(Path(output_dir) / "audio_files"),
            limit=limit,
        )
        logger.info(f"Extracted {len(audio_files)} audio files")

    except Exception as err:
        logger.error(f"Error processing audio data: {err}")
        raise RuntimeError("Failed to process audio data") from err


if __name__ == "__main__":
    process_audio_data()
