from pathlib import Path
from typing import Optional

from datasetplus import AudioProcessor, HFDatasetManager
from datasetplus.utils import get_logger

logger = get_logger(__name__)


def process_audio_data(
    dataset_name: str = "bookbot/ljspeech_phonemes",
    output_dir: str = "output/ljspeech_phonemes",
    columns_to_extract: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> None:
    """Process audio data from a Hugging Face dataset.

    Args:
        dataset_name: Name of the Hugging Face dataset.
        output_dir: Directory to store the output.
        columns_to_extract: List of column names to extract (besides 'audio').
        limit: Maximum number of files to process. Defaults to None (all files).
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
        data_dir = Path(output_dir) / "data"
        processor = AudioProcessor(data_dir)

        # Get metadata and print sample data
        metadata = processor.get_metadata()
        logger.info(f"Dataset metadata: {metadata}")
        
        # Print column names and first row for debugging
        df = processor.df
        logger.info(f"Column names: {df.columns}")
        logger.info(f"First row: {df.head(1)}")

        # Extract audio files
        audio_files = processor.extract_audio_files(
            output_dir=str(Path(output_dir) / "audio_files"),
            columns_to_extract=columns_to_extract,
            limit=limit,
        )
        logger.info(f"Extracted {len(audio_files)} audio files")

    except Exception as err:
        logger.error(f"Error processing audio data: {err}")
        raise RuntimeError("Failed to process audio data") from err


if __name__ == "__main__":
    process_audio_data(
        dataset_name="bookbot/ljspeech_phonemes",
        output_dir="output/ljspeech_phonemes",
        columns_to_extract=["text", "phonemes"],
        limit=None,
    )
