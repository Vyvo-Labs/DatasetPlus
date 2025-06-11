"""
Simple utility function for downloading audio files from Hugging Face datasets.
Loads a dataset and downloads the audio files in WAV format.
"""

import os
from typing import List, Optional
from tqdm.auto import tqdm
import soundfile as sf

# Import datasets library
from datasets import load_dataset, Audio


def download_audio_files(
    dataset_name: str,
    output_dir: str,
    split: str = "train",
    audio_column: str = "audio",
    num_samples: Optional[int] = None,
    id_column: Optional[str] = None,
) -> List[str]:
    """
    Download audio files from a Hugging Face dataset and save them as WAV files.

    Args:
        dataset_name: Name of the dataset on Hugging Face (e.g., 'OpenSpeechHubCAVA/2M-Belebele-Ja')
        output_dir: Directory to save the downloaded audio files
        split: Dataset split to download (e.g., 'train', 'validation', 'test')
        audio_column: Name of the column containing audio data
        num_samples: If provided, only download this many samples (useful for testing)
        id_column: Column to use as the filename prefix. If None, will use index numbers.

    Returns:
        List of paths to the saved audio files
    """
    # Simple implementation

    print(f"Loading dataset: {dataset_name}, split: {split}")

    try:
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        # Take a sample if requested
        if num_samples is not None and num_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
            print(f"Selected {num_samples} samples from the dataset")

        # Ensure dataset has audio column
        if audio_column not in dataset.column_names:
            raise ValueError(
                f"Audio column '{audio_column}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        # Cast to Audio feature if needed
        if not isinstance(dataset.features[audio_column], Audio):
            print(f"Converting {audio_column} column to Audio feature...")
            dataset = dataset.cast_column(audio_column, Audio())

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download and save audio files
        saved_files = []
        for i, item in enumerate(tqdm(dataset, desc="Downloading audio files")):
            try:
                # Get audio data
                audio_data = item[audio_column]

                # Generate file name
                if id_column and id_column in item:
                    # Use the provided column as file name prefix
                    file_id = str(item[id_column])
                    # Clean up the ID to make a valid filename
                    file_id = "".join(c if c.isalnum() else "_" for c in file_id)
                else:
                    # Use index as file name
                    file_id = f"audio_{i:05d}"

                file_path = os.path.join(output_dir, f"{file_id}.wav")

                # Save audio file
                if (
                    isinstance(audio_data, dict)
                    and "array" in audio_data
                    and "sampling_rate" in audio_data
                ):
                    # Handle array format (most common from datasets)
                    sf.write(
                        file_path, audio_data["array"], audio_data["sampling_rate"]
                    )
                    saved_files.append(file_path)
                elif isinstance(audio_data, dict) and "path" in audio_data:
                    # Handle path format (copy file)
                    import shutil

                    shutil.copy(audio_data["path"], file_path)
                    saved_files.append(file_path)
                else:
                    print(f"Unsupported audio format for item {i}, skipping")

            except Exception as e:
                print(f"Error processing audio item {i}: {str(e)}")

        # No metadata file needed

        print(f"\nDownloaded {len(saved_files)} audio files to {output_dir}")
        return saved_files

    except Exception as e:
        print(f"Error downloading dataset {dataset_name}: {str(e)}")
        return []
