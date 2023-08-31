import glob
import logging
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    IO,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)

import boto3
import torch
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

IDType = TypeVar("IDType")


class BaseStorageProvider(Generic[IDType], ABC):
    """Base class for storage providers.

    Args:
        id_to_key (Callable, optional): Function to map a file ID to a storage key.
                                        Defaults to None.
        key_to_id (Callable, optional): Function to map a storage key to a file ID.
                                        Defaults to None.

    Attributes:
        file_ids (List[IDType]): List of file IDs in the storage provider.

    """

    def __init__(
        self,
        id_to_key: Optional[Callable[[IDType], str]] = None,
        key_to_id: Optional[Callable[[str], IDType]] = None,
        device: str = "cpu",
    ):
        self._id_to_key = id_to_key or self.default_id_to_key
        self._key_to_id = key_to_id or self.default_key_to_id
        self.file_ids: List[IDType] = []
        self.device = torch.device(device)

    @abstractmethod
    def save_file(self, file_id: IDType, file: Any):
        """Abstract method to save a file."""
        raise NotImplementedError

    @abstractmethod
    def load_file(self, file_id: IDType):
        """Abstract method to load a file."""
        raise NotImplementedError

    @abstractmethod
    def get_file_ids(self) -> List[IDType]:
        """Abstract method to get a list of file IDs."""
        raise NotImplementedError

    def id_to_key(self, file_id: IDType) -> str:
        """Map a file ID to a storage key."""
        return self._id_to_key(file_id)

    def key_to_id(self, key: str) -> IDType:
        """Map a storage key to a file ID."""
        return self.default_key_to_id(key)

    @staticmethod
    def default_id_to_key(file_id: IDType) -> str:
        """Default method to map a file ID to a storage key."""
        return f"{file_id}.pt"

    @staticmethod
    def default_key_to_id(key: str) -> IDType:
        """Default method to map a storage key to a file ID."""
        warnings.warn(
            "Using default key_to_id. This yields a string, which may not be what you want."
        )
        return key.replace(".pt", "")  # type: ignore

    def __iter__(self):
        for file_id in self.file_ids:
            yield self.load_file(file_id)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx: Union[int, IDType]):
        if isinstance(idx, int):
            return self.load_file(self.file_ids[idx])

        elif idx not in self.file_ids:
            warnings.warn(f"File with id `{idx}` not found.")
            return self.load_file(idx)

        raise TypeError(f"Invalid argument `{idx}` of type `{type(idx)}`")

    def __contains__(self, file_id):
        return file_id in self.file_ids


class LocalStorageProvider(BaseStorageProvider):
    """Local storage provider.

    Args:
        local_root (str): Base directory in which to save files locally.
        id_to_key (Callable): Function to map a file ID to a storage key.
        key_to_id (Callable): Function to map a storage key to a file ID.
    """

    def __init__(
        self,
        local_root: str,
        id_to_key: Optional[Callable[[IDType], str]] = None,
        key_to_id: Optional[Callable[[str], IDType]] = None,
        device: str = "cpu",
    ):
        super().__init__(id_to_key, key_to_id, device=device)
        self.local_root = Path(local_root)

    def save_file(self, file_id: IDType, file: Any):
        """Save a file locally."""
        key = self.id_to_key(file_id)
        rel_file_path = self.local_root / key
        torch.save(file, rel_file_path)

    def load_file(self, file_id: IDType):
        """Load a file locally."""
        key = self.id_to_key(file_id)
        rel_file_path = self.local_root / key
        return torch.load(rel_file_path)

    def get_file_ids(self) -> List[IDType]:
        """
        Returns a list of tuples of all files in the local directory.
        """
        files = glob.glob(f"{self.local_root}/{self.id_to_key('*')}")
        return sorted([self.name_to_id(os.path.basename(f)) for f in files])  # type: ignore

    def __repr__(self):
        return f"LocalStorageProvider({self.local_root})"


class S3StorageProvider(BaseStorageProvider[IDType]):
    """AWS S3 Storage Provider.

    Args:
        bucket_name (str): Name of the S3 bucket.
        id_to_key (Callable): Function to map a file ID to a storage key.
        key_to_id (Callable): Function to map a storage key to a file ID.
        tmp_dir (str, optional): Temporary directory for file operations. Defaults to "/tmp".
    """

    def __init__(
        self,
        bucket_name: str,
        id_to_key: Optional[Callable[[IDType], str]] = None,
        key_to_id: Optional[Callable[[str], IDType]] = None,
        tmp_dir: str = "/tmp",
        device: str = "cpu",
    ):
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            raise EnvironmentError("AWS environment variables not set.")

        super().__init__(id_to_key, key_to_id, device=device)
        self.client = boto3.client("s3")
        self.bucket_name = bucket_name
        self.tmp_dir = Path(tmp_dir)

    def save_file(self, file_id: IDType, file: Any):
        """Save a file to an S3 bucket."""
        temp_path = self.tmp_dir / self.id_to_key(file_id)
        torch.save(file, temp_path)
        self.client.upload_file(temp_path, self.bucket_name, self.id_to_key(file_id))

    def load_file(self, file_id: IDType):
        """Load a file from an S3 bucket."""
        temp_path = self.tmp_dir / self.id_to_key(file_id)
        self.client.download_file(self.bucket_name, self.id_to_key(file_id), temp_path)
        return torch.load(temp_path, map_location=self.device)

    def get_file_ids(self) -> List[IDType]:
        """
        Returns a list of tuples of all files in the bucket directory.
        """

        response = self.client.list_objects_v2(Bucket=self.bucket_name)
        if "Contents" in response:
            return sorted(
                [
                    self.key_to_id(item["Key"])
                    for item in response["Contents"]
                    if item["Key"].startswith(
                        self.parent_dir
                    )  # TODO: Filter for the right files
                ]
            )

        return []

    def __repr__(self):
        return f"S3StorageProvider({self.bucket_name}, {self.tmp_dir})"


class CompositeStorageProvider(BaseStorageProvider[IDType]):
    """Composite storage provider that can use multiple providers.

    Args:
        providers (List[BaseStorageProvider]): List of storage providers to use.
    """

    def __init__(self, providers: List[BaseStorageProvider[IDType]]):
        self.providers = providers

    def save_file(self, file_id: IDType, file: Any):
        """Save a file using all the underlying storage providers."""
        for provider in self.providers:
            provider.save_file(file_id, file)

    def load_file(self, file_id: IDType):
        """Load a file from one of the underlying storage providers."""
        for provider in self.providers:
            try:
                return provider.load_file(file_id)
            except FileNotFoundError:
                continue
        raise FileNotFoundError("File not found in any provider")

    def get_file_ids(self) -> List[IDType]:
        """
        Returns a list of tuples of all files in the bucket directory.
        """

        file_ids = set()

        for provider in self.providers:
            file_ids |= set(provider.get_file_ids())

        return sorted(list(file_ids))

    def __repr__(self):
        return f"CompositeStorageProvider({self.providers})"


# TODO: Rename the following
def StorageProvider(
    bucket_name: Optional[str] = None,
    local_root: Optional[str] = None,
    parent_dir: str = "data",
    device="cpu",
):
    """A factory for creating composite storage providers and syncing them with a shared temp directory."""
    shared_temp_dir = local_root or "/tmp"

    def create_provider(
        provider_type: str, **kwargs
    ) -> Union[S3StorageProvider, LocalStorageProvider]:
        if provider_type == "s3":
            return S3StorageProvider(tmp_dir=shared_temp_dir, **kwargs)
        elif provider_type == "local":
            return LocalStorageProvider(local_root=shared_temp_dir, **kwargs)
        else:
            raise ValueError("Invalid provider_type.")

    def create_composite_provider(
        types_and_configs: List[Tuple[str, dict]]
    ) -> CompositeStorageProvider:
        providers = [
            create_provider(t, device=device, **c) for t, c in types_and_configs
        ]
        return CompositeStorageProvider(providers)

    providers = []

    if bucket_name:
        providers.append(("s3", {"bucket_name": bucket_name, "parent_dir": parent_dir}))

    if local_root:
        providers.append(
            ("local", {"local_root": local_root, "parent_dir": parent_dir})
        )

    return create_composite_provider(providers)


class StorageProvider(Generic[IDType], ABC):
    """
    Wrapper for either local or cloud (S3) storage (or both).

    :param bucket_name: The name of the S3 bucket to store checkpoints (Optional)
    :param local_root: If provided, then the base directory in which to save files locally. If omitted, files will not be saved locally. (Optional)
    :param save_locally: If True, saves checkpoints locally without deleting them (Optional)
    """

    file_ids: List[IDType]

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        local_root: Optional[str] = None,
        parent_dir: str = "data",
        device=torch.device("cpu"),
    ):
        self.bucket_name = bucket_name
        self.is_local_enabled = local_root is not None
        self.local_root = Path(local_root or "tmp")
        self.parent_dir = parent_dir
        self.device = device

        self.file_ids = []  # Any non-int hashable type.
        self.client = None

        # Cloud

        # Local

        local_path = os.path.join(self.local_root, parent_dir)
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        if not self.bucket_name and not self.local_root:
            warnings.warn(
                "Neither S3 bucket name provided nor local_root is defined. Files will not be persisted."
            )

    def id_to_name(self, file_id: Union[IDType, Literal["*"]]) -> str:
        """Should contain no `/` and should handle the wildcard."""
        raise NotImplementedError

    def id_to_key(self, file_id: Union[IDType, Literal["*"]]) -> str:
        return f"{self.parent_dir}/{self.id_to_name(file_id)}.pt"

        checkpoint = torch.load(rel_file_path, map_location=self.device)

        if not self.is_local_enabled and self.bucket_name and self.client:
            os.remove(rel_file_path)

        return checkpoint


EpochAndBatch = Tuple[int, int]


class CheckpointManager(StorageProvider[EpochAndBatch]):
    def __init__(
        self,
        project_dir: str,
        bucket_name: Optional[str] = None,
        local_root: Optional[str] = None,
        device=torch.device("cpu"),
    ):
        super().__init__(
            bucket_name, local_root, f"checkpoints/{project_dir}", device=device
        )

    @staticmethod
    def id_to_name(file_id: Union[EpochAndBatch, Literal["*"]]) -> str:
        if file_id == "*":
            return "*"

        epoch, batch = file_id
        return f"checkpoint_epoch_{epoch}_batch_{batch}"

    @staticmethod
    def name_to_id(name: str) -> EpochAndBatch:
        parts = name.split("_")
        epoch = int(parts[-3])
        batch_idx = int(parts[-1].split(".")[0])
        return epoch, batch_idx

    def __repr__(self):
        return f"CheckpointManager({self.parent_dir}, {self.bucket_name})"
