"""Audio processing utilities for dataset management.

This module provides the AudioProcessor class for handling audio data stored in
Parquet files, including metadata extraction and file processing.
"""

from pathlib import Path
from typing import Optional, Union

import polars as pl
from tqdm import tqdm

from .utils import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """A class for processing audio data from Parquet files."""

    def __init__(self, parquet_path: Union[str, Path]) -> None:
        """Initialize AudioProcessor.

        Args:
            parquet_path: Path to the parquet file containing audio data
        """
        self.parquet_path = Path(parquet_path)
        self._df: Optional[pl.DataFrame] = None

    @property
    def df(self) -> pl.DataFrame:
        """Lazy load the parquet file.

        Returns:
            pl.DataFrame: Polars DataFrame containing the audio data.

        Raises:
            ValueError: If the parquet file cannot be read or is invalid.
        """
        if self._df is None:
            try:
                logger.info(f"Loading parquet file: {self.parquet_path}")
                self._df = pl.read_parquet(self.parquet_path)
            except Exception as err:
                logger.error(f"Failed to read parquet file: {err}")
                raise ValueError("Failed to read parquet file") from err
        return self._df

    def extract_audio_files(
        self, output_dir: Union[str, Path], limit: Optional[int] = None
    ) -> list[Path]:
        """Extract audio files from the parquet file.

        Args:
            output_dir: Directory to save the extracted audio files
            limit: Maximum number of files to extract (None for all)

        Returns:
            list[Path]: List of paths to the extracted audio files

        Raises:
            ValueError: If audio files cannot be extracted
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = self.df
            if limit is not None:
                df = df.head(limit)

            audio_files = []
            total_files = len(df)
            progress = tqdm(
                df.iter_rows(),
                total=total_files,
                desc="Extracting audio files",
            )

            for row in progress:
                audio_path = output_dir / f"{row[0]}.wav"
                with open(audio_path, "wb") as f:
                    f.write(row[1])
                audio_files.append(audio_path)
                logger.debug(f"Extracted: {audio_path}")

            logger.info(f"Extracted {len(audio_files)} audio files")
            return audio_files

        except Exception as err:
            logger.error(f"Failed to extract audio files: {err}")
            raise ValueError("Failed to extract audio files") from err

    def get_metadata(self) -> dict[str, Union[int, list[str], pl.DataFrame]]:
        """Get basic metadata about the dataset.

        Returns:
            Dict containing metadata with the following keys:
                - total_files (int): Total number of files in the dataset
                - file_size_mb (float): Total size in MB
                - columns (List[str]): List of column names
                - sample (pl.DataFrame): Sample of the first few rows

        Raises:
            ValueError: If metadata cannot be retrieved.
        """
        try:
            df = self.df
            total_size = sum(len(row[1]) for row in df.iter_rows())
            metadata = {
                "total_files": len(df),
                "file_size_mb": total_size / (1024 * 1024),
                "columns": df.columns,
                "sample": df.head(5),
            }
            logger.debug(f"Retrieved metadata: {metadata}")
            return metadata

        except Exception as err:
            logger.error(f"Failed to get metadata: {err}")
            raise ValueError("Failed to get metadata") from err
