import shutil
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from datasetplus.utils.logger import get_logger

logger = get_logger(__name__)


class HFDatasetManager:
    """A class for managing Hugging Face datasets.

    This class provides functionality to download and manage datasets from the
    Hugging Face Hub.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize HFDatasetManager.

        Args:
            token: Hugging Face authentication token
        """
        self.token = token
        self.api = HfApi(token=token)

    def download(
        self,
        repo_id: str,
        local_dir: Union[str, Path],
        filename: Optional[str] = None,
        repo_type: str = "dataset",
        ignore_patterns: Optional[list[str]] = None,
        no_cache: bool = False,
    ) -> Path:
        """Download a dataset from Hugging Face Hub.

        Args:
            repo_id: Name of the repository (e.g., 'username/dataset-name')
            local_dir: Local directory to save the dataset
            filename: Specific file to download (optional)
            repo_type: Type of repository ('dataset' or 'model')
            ignore_patterns: List of file patterns to ignore
            no_cache: If True, force download even if files exist in cache

        Returns:
            Path: Path to the downloaded dataset directory
        """
        try:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)

            # Varsayılan olarak yok sayılacak dosya ve dizinler
            default_ignore = [
                ".git*",
                "README.md",
                "*.md",
                "LICENSE",
                "__pycache__",
                "*.pyc",
                ".DS_Store"
            ]
            
            # Kullanıcının belirttiği ignore_patterns'i de ekle
            if ignore_patterns:
                default_ignore.extend(ignore_patterns)

            if filename:
                # Tek bir dosya indir
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    token=self.token,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                )
                logger.info(f"Downloaded file: {file_path}")
                return Path(file_path)
            else:
                # Tüm repository'yi indir
                dataset_path = snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=self.token,
                    local_dir=local_dir,
                    ignore_patterns=default_ignore,
                    local_dir_use_symlinks=False,
                )
                
                # .cache klasörünü sil
                cache_dir = Path(local_dir) / ".cache"
                if cache_dir.exists():
                    logger.info(f"Removing cache directory: {cache_dir}")
                    shutil.rmtree(cache_dir)
                
                logger.info(f"Downloaded dataset to: {dataset_path}")
                return Path(dataset_path)

        except Exception as err:
            logger.error(f"Failed to download dataset: {err}")
            raise RuntimeError("Failed to download dataset") from err

    def upload(
        self,
        local_path: Union[str, Path],
        repo_id: str,
        repo_type: str = "dataset",
    ) -> None:
        """Upload a dataset to Hugging Face Hub.

        Args:
            local_path: Path to the local dataset
            repo_id: Name of the repository to upload to
            repo_type: Type of repository ('dataset' or 'model')
        """
        try:
            local_path = Path(local_path)
            logger.info(f"Uploading {local_path} to {repo_id}")

            if local_path.is_file():
                # Upload single file
                self.api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=local_path.name,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
            elif local_path.is_dir():
                # Upload entire directory
                self.api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    repo_type=repo_type,
                )

            logger.info("Upload complete")
        except Exception as e:
            logger.error(f"Failed to upload to {repo_id}: {str(e)}")
            raise RuntimeError(f"Failed to upload to {repo_id}") from e
