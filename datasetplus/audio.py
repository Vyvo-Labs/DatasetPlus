import polars as pl
from pathlib import Path
from typing import Optional, List, Union
from .utils import logger

class AudioProcessor:
    """Process audio data from parquet files.
    
    This class provides functionality to handle audio data stored in parquet files,
    including extraction of audio files and metadata retrieval.
    
    Attributes:
        parquet_path (Path): Path to the parquet file containing audio data
        
    Example:
        >>> processor = AudioProcessor("path/to/data.parquet")
        >>> metadata = processor.get_metadata()
        >>> audio_files = processor.extract_audio_files("output_dir", limit=5)
    """
    
    def __init__(self, parquet_path: Union[str, Path]):
        """Initialize AudioProcessor with a parquet file path.
        
        Args:
            parquet_path (Union[str, Path]): Path to the parquet file containing audio data.
                The parquet file should have a column containing audio data in bytes format.
        
        Raises:
            FileNotFoundError: If the parquet file does not exist.
            ValueError: If the parquet file is invalid or doesn't contain audio data.
        """
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
        self._df = None
    
    @property
    def df(self) -> pl.DataFrame:
        """Lazy load the parquet file.
        
        Returns:
            pl.DataFrame: Polars DataFrame containing the audio data.
            
        Raises:
            ValueError: If the parquet file cannot be read or is invalid.
        """
        if self._df is None:
            logger.info(f"Loading parquet file: {self.parquet_path}")
            try:
                self._df = pl.read_parquet(self.parquet_path)
            except Exception as e:
                logger.error(f"Failed to read parquet file: {e}")
                raise ValueError(f"Failed to read parquet file: {e}")
        return self._df
    
    def extract_audio_files(self, output_dir: Union[str, Path], limit: Optional[int] = None) -> List[Path]:
        """Extract audio files from the parquet file.
        
        Args:
            output_dir (Union[str, Path]): Directory to save the audio files.
                Will be created if it doesn't exist.
            limit (Optional[int], optional): Maximum number of files to extract.
                If None, extracts all files. Defaults to None.
            
        Returns:
            List[Path]: List of paths to the extracted audio files.
            
        Raises:
            ValueError: If the audio data cannot be extracted or saved.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        total_files = min(len(self.df), limit) if limit else len(self.df)
        
        logger.info(f"Extracting {total_files} audio files to {output_dir}")
        
        try:
            for i, row in enumerate(self.df.iter_rows()):
                if limit and i >= limit:
                    break
                    
                # Get the audio data from the struct column
                audio_struct = row[2]  # Index 2 is the audio column
                audio_bytes = audio_struct['bytes']
                
                # Save the audio file
                audio_path = output_dir / f"audio_{i}.wav"
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                saved_files.append(audio_path)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Extracted {i + 1}/{total_files} files")
            
            logger.info(f"Successfully extracted {len(saved_files)} audio files")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to extract audio files: {e}")
            raise ValueError(f"Failed to extract audio files: {e}")
    
    def get_metadata(self) -> dict:
        """Get basic metadata about the dataset.
        
        Returns:
            dict: Dictionary containing metadata with the following keys:
                - total_files (int): Total number of files in the dataset
                - schema (pl.Schema): Schema of the parquet file
                - columns (List[str]): List of column names
                - sample (pl.DataFrame): Sample of the first few rows
                
        Raises:
            ValueError: If metadata cannot be retrieved.
        """
        try:
            metadata = {
                "total_files": len(self.df),
                "schema": self.df.schema,
                "columns": self.df.columns,
                "sample": self.df.head()
            }
            logger.debug("Retrieved dataset metadata successfully")
            return metadata
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            raise ValueError(f"Failed to get metadata: {e}")
