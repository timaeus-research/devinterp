import glob
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union, Generic, TypeVar, Literal, Set

import boto3
import torch
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

IDType = TypeVar("IDType")

class StorageProvider(Generic[IDType]):
    """
    Wrapper for either local or cloud (S3) storage (or both).

    :param bucket_name: The name of the S3 bucket to store checkpoints (Optional)
    :param local_root: If provided, then the base directory in which to save files locally. If omitted, files will not be saved locally. (Optional)
    :param save_locally: If True, saves checkpoints locally without deleting them (Optional)
    """
    file_ids: List[IDType]

    def __init__(self, bucket_name: Optional[str] = None, local_root: Optional[str] = None, parent_dir: str = "data",  device=torch.device("cpu")):
        self.bucket_name = bucket_name
        self.is_local_enabled = local_root is not None
        self.local_root = Path(local_root or "tmp")
        self.parent_dir = parent_dir
        self.device = device

        self.file_ids = [] # Any non-int hashable type.
        self.client = None

        # Cloud 

        if bucket_name and (os.getenv("AWS_SECRET_ACCESS_KEY") and os.getenv("AWS_ACCESS_KEY_ID")):
            self.client = boto3.client("s3")
            self.file_ids = self.get_file_ids()
        else:
            warnings.warn("AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID must be set to use S3 bucket.")

        # Local 
        
        local_path = os.path.join(self.local_root, parent_dir)
        if self.is_local_enabled and not os.path.exists(local_path):
            os.makedirs(local_path)
        
        if not self.bucket_name and not self.local_root:
            warnings.warn("Neither S3 bucket name provided nor local_root is defined. Files will not be persisted.")

    @property
    def is_s3_enabled(self):
        return self.client is not None
    
    def id_to_name(self, file_id: Union[IDType, Literal["*"]]) -> str:
        """Should contain no `/` and should handle the wildcard."""
        raise NotImplementedError
    
    def id_to_key(self, file_id: Union[IDType, Literal["*"]]) -> str:
        return f"{self.parent_dir}/{self.id_to_name(file_id)}.pt"
    
    def name_to_id(self, name: str) -> IDType:
        raise NotImplementedError

    def get_file_ids(self) -> List[IDType]:
        """
        Returns a list of tuples (epoch, batch_idx) of all checkpoints in the bucket or local directory.
        """
        file_ids: Set[IDType] = set()

        if self.is_local_enabled:
            checkpoint_files = glob.glob(self.local_root / self.id_to_key("*"))
            file_ids |= {self.name_to_id(os.path.basename(f)) for f in checkpoint_files}

        if self.is_s3_enabled
            response = self.client.list_objects_v2(Bucket=self.bucket_name)
            if "Contents" in response:
                file_ids |= {self.name_to_id(item["Key"]) for item in response["Contents"] if item["Key"].startswith(self.parent_dir)}
            
        return sorted(list(file_ids))

    def upload_file(self, file_path: str, key: str):
        self.client.upload_file(file_path, self.bucket_name, key)

    def save_file(self, file_id: str, file):
        file_path = self.id_to_name(file_id)
        rel_file_path = self.local_root / file_path
        torch.save(file, rel_file_path)

        if self.client:
            self.upload_file(rel_file_path, file_path)

        if not self.is_local_enabled:
            os.remove(rel_file_path)

    def load_file(self, file_id):
        file_path = self.id_to_name(file_id)
        rel_file_path = self.local_root / file_path

        if (self.is_local_enabled and os.path.exists(rel_file_path)):
            logger.info(f"Loading {file_path} from local save...")
        elif self.client:
            logger.info(f"Downloading {file_path} from bucket `{self.bucket_name}`...")
            self.client.download_file(self.bucket_name, file_path, rel_file_path)
        else:
            raise OSError(f"File with id `{file_id}` not found either locally or in bucket.")

        checkpoint = torch.load(rel_file_path, map_location=self.device)

        if not self.is_local_enabled and self.bucket_name and self.client:
            os.remove(rel_file_path)

        return checkpoint

    def __iter__(self):
        for file_id in self.file_ids:
            yield self.load_file(file_id)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.load_file(self.file_ids[idx])
        
        elif idx not in self.file_ids:
            warnings.warn(f"File with id `{idx}` not found in {self.bucket_name}.")
            return self.load_file(idx)

        raise TypeError(f"Invalid argument `{idx}` of type `{type(idx)}`")


    def __contains__(self, file_id):
        return file_id in self.file_ids

    def __repr__(self):
        return f"StorageProvider({self.bucket_name}, {self.local_root})"

EpochAndBatch = Tuple[int, int]

class CheckpointManager(StorageProvider[EpochAndBatch]):
    @staticmethod
    def id_to_name(id: Union[EpochAndBatch, Literal["*"]]) -> str:
        if id == "*":
            return "*"
        
        epoch, batch = id
        return f"checkpoint_epoch_{epoch}_batch_{batch}"

    @staticmethod
    def name_to_id(name: str) -> EpochAndBatch:
        parts = name.split("_")
        epoch = int(parts[-3])
        batch_idx = int(parts[-1].split(".")[0])
        return epoch, batch_idx

    def __repr__(self):
        return f"CheckpointManager({self.parent_dir}, {self.bucket_name})"


NeuronSeedBatch = Tuple[int, int, int]

class VisualizationManager(StorageProvider):
    @staticmethod
    def id_to_name(id: NeuronSeedBatch):
        if id == "*":
            return "*"
        
        neuron, seed, batch = id
        return f"visualization_neuron_{neuron}_seed_{seed}_batch_{batch}"

    @staticmethod
    def name_to_id(name: str) -> NeuronSeedBatch:
        parts = name.split("_")
        neuron = int(parts[-5])
        seed = int(parts[-3])
        batch_idx = int(parts[-1].split(".")[0])
        return neuron, seed, batch_idx

    def __repr__(self):
        return f"VisualizationManager({self.parent_dir}, {self.bucket_name})"