from pathlib import Path
from typing import Optional, Union

import polars as pl
from tqdm import tqdm

from datasetplus.utils.logger import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """A class for processing audio data from Parquet files."""

    def __init__(self, parquet_path: Union[str, Path]) -> None:
        """Initialize AudioProcessor.

        Args:
            parquet_path: Path to the parquet file or directory containing parquet files
        """
        self.parquet_path = Path(parquet_path)
        self._df: Optional[pl.DataFrame] = None

    @property
    def df(self) -> pl.DataFrame:
        """Lazy load the parquet file(s).

        Returns:
            pl.DataFrame: Polars DataFrame containing the audio data.

        Raises:
            ValueError: If the parquet file(s) cannot be read or is invalid.
        """
        if self._df is None:
            try:
                if self.parquet_path.is_file():
                    logger.info(f"Loading single parquet file: {self.parquet_path}")
                    self._df = pl.read_parquet(self.parquet_path)
                else:
                    logger.info(f"Loading parquet files from directory: {self.parquet_path}")
                    parquet_files = list(self.parquet_path.glob("*.parquet"))
                    if not parquet_files:
                        raise ValueError(f"No parquet files found in {self.parquet_path}")

                    # Read and concatenate all parquet files
                    dfs = []
                    for file in parquet_files:
                        logger.info(f"Loading parquet file: {file}")
                        dfs.append(pl.read_parquet(file))
                    self._df = pl.concat(dfs)

            except Exception as err:
                logger.error(f"Failed to read parquet file(s): {err}")
                raise ValueError("Failed to read parquet file(s)") from err
        return self._df

    def extract_audio_files(
        self,
        output_dir: Union[str, Path],
        columns_to_extract: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[Path]:
        """Extract audio files and their corresponding text from the parquet file.

        Args:
            output_dir: Directory to save the extracted audio files and text files
            columns_to_extract: List of column names to extract (besides 'audio').
                              If None, all non-audio columns will be extracted.
            limit: Maximum number of files to extract (None for all)

        Returns:
            list[Path]: List of paths to the extracted audio files

        Raises:
            ValueError: If files cannot be extracted
        """
        try:
            output_dir = Path(output_dir)
            audio_dir = output_dir / "wavs"
            audio_dir.mkdir(parents=True, exist_ok=True)

            df = self.df
            available_columns = set(df.columns)
            logger.info(f"Available columns in dataset: {available_columns}")

            if columns_to_extract is None:
                columns_to_extract = [
                    col for col in available_columns if col != "audio" and not col.startswith("__")
                ]
            else:
                invalid_columns = set(columns_to_extract) - available_columns
                if invalid_columns:
                    raise ValueError(f"Column(s) not found in dataset: {invalid_columns}")

            logger.info(f"Columns to extract: {columns_to_extract}")

            column_dirs = {}
            for col in columns_to_extract:
                col_dir = output_dir / col
                col_dir.mkdir(parents=True, exist_ok=True)
                column_dirs[col] = col_dir
                logger.info(f"Created directory for {col}: {col_dir}")

            if limit is not None:
                df = df.head(limit)

            total_files = len(df)
            progress = tqdm(
                df.iter_rows(named=True),
                total=total_files,
                desc="Extracting files",
            )

            audio_files: list[Path] = []
            for row in progress:
                audio_data = row.get("audio", {})
                if isinstance(audio_data, dict):
                    audio_bytes = audio_data.get("bytes", b"")
                else:
                    audio_bytes = audio_data

                filename = str(row.get("id", row.get("filename", len(audio_files))))

                audio_path = audio_dir / f"{filename}.wav"
                if isinstance(audio_bytes, bytes):
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    audio_files.append(audio_path)
                    logger.debug(f"Extracted audio: {audio_path}")

                    for col, col_dir in column_dirs.items():
                        content = row.get(col, "")
                        if content:
                            file_path = col_dir / f"{filename}.txt"
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(str(content))
                            logger.debug(f"Extracted {col}: {file_path}")
                else:
                    logger.warning(f"Skipping invalid audio data for {filename}")

            logger.info(f"Extracted {len(audio_files)} files to:")
            logger.info(f"  - Audio files: {audio_dir}")
            for col, col_dir in column_dirs.items():
                logger.info(f"  - {col} files: {col_dir}")
            return audio_files

        except Exception as err:
            logger.error(f"Failed to extract files: {err}")
            raise ValueError("Failed to extract files") from err

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
