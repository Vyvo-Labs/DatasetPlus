from datasetplus import HFDatasetManager, AudioProcessor
from datasetplus.utils import logger
from pathlib import Path

def process_audio_data(
    dataset_name: str = "fixie-ai/llama-questions",
    output_dir: str = "output/llama_questions",
    limit: int = 5
) -> None:
    """Process audio data from a Hugging Face dataset."""
    try:
        # Initialize and download dataset
        dataset = HFDatasetManager()
        output_path = Path(output_dir)
        
        logger.info(f"Processing dataset: {dataset_name}")
        dataset.download(
            repo_id=dataset_name,
            local_dir=output_path,
            repo_type="dataset",
            ignore_patterns=[".gitignore", "README.md", ".gitattributes"]
        )
        
        # Process audio data
        parquet_files = list(output_path.glob("**/*.parquet"))
        if not parquet_files:
            logger.error("No parquet files found in the dataset")
            return
        
        processor = AudioProcessor(parquet_files[0])
        metadata = processor.get_metadata()
        logger.info(f"Processing {metadata['total_files']} audio files")
        
        # Extract audio files
        audio_dir = output_path / "audio_files"
        processor.extract_audio_files(audio_dir, limit=limit)
            
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        raise

if __name__ == "__main__":
    process_audio_data(
        dataset_name="fixie-ai/llama-questions",
        output_dir="output/llama_questions",
        limit=None  # Set to None to extract all audio files
    )
