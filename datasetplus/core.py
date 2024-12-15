from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path
import shutil
from tqdm import tqdm
from .utils import logger

class HFDatasetManager:
    """A class for managing datasets on Hugging Face Hub."""

    def __init__(self, token=None):
        """Initialize HFDatasetManager.

        Args:
            token (str, optional): Hugging Face API token. Defaults to None.
        """
        self.api = HfApi(token=token)
        self.token = token
        logger.info(f"Initialized HFDatasetManager with token: {token}")

    def download(self, repo_id, local_dir, filename=None, repo_type="dataset", ignore_patterns=None, no_cache=False):
        """Download a dataset from Hugging Face Hub.

        Args:
            repo_id (str): Repository ID on Hugging Face Hub.
            local_dir (str): Local directory to save the dataset.
            filename (str, optional): Specific file to download. Defaults to None.
            repo_type (str, optional): Repository type. Defaults to "dataset".
            ignore_patterns (list, optional): List of patterns to ignore. Defaults to None.
            no_cache (bool, optional): If True, disable caching. Defaults to False.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = local_dir / ".cache"

        logger.info(f"Downloading dataset from {repo_id} to {local_dir}")

        if filename:
            # Download specific file
            if ignore_patterns and any(pattern in filename for pattern in ignore_patterns):
                logger.info(f"Ignoring file: {filename}")
                return
            logger.info(f"Downloading file: {filename}")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=self.token,
                local_dir=local_dir,
                repo_type=repo_type
            )
            logger.debug(f"Successfully downloaded {filename}")
        else:
            # Download all files except ignored ones
            files = self.api.list_repo_files(repo_id, repo_type=repo_type)
            if ignore_patterns:
                files = [f for f in files if not any(pattern in f for pattern in ignore_patterns)]
            
            logger.info(f"Found {len(files)} files to download")
            for file in tqdm(files, desc="Downloading files"):
                logger.debug(f"Downloading: {file}")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    token=self.token,
                    local_dir=local_dir,
                    repo_type=repo_type
                )
        
        # Delete .cache folder if it exists
        if cache_dir.exists():
            logger.info(f"Cleaning up cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            logger.debug(f"Removed cache directory: {cache_dir}")

    def upload(self, local_path, repo_id, repo_type="dataset"):
        """Upload a dataset to Hugging Face Hub.

        Args:
            local_path (str): Local path to the dataset.
            repo_id (str): Repository ID on Hugging Face Hub.
            repo_type (str, optional): Repository type. Defaults to "dataset".
        """
        logger.info(f"Uploading dataset from {local_path} to {repo_id}")
        
        try:
            # Create the repository if it doesn't exist
            self.api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
            logger.debug(f"Repository {repo_id} is ready")

            local_path = Path(local_path)
            if local_path.is_file():
                # Upload single file
                self.api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=local_path.name,
                    repo_id=repo_id,
                    repo_type=repo_type
                )
            elif local_path.is_dir():
                # Upload directory
                self.api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
            else:
                raise ValueError(f"Path {local_path} does not exist")
            logger.info(f"Successfully uploaded dataset to {repo_id}")
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            raise
