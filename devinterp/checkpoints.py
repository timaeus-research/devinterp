import glob
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import boto3
import torch
from botocore.exceptions import ClientError


class CheckpointManager:
    """
    Manages the saving and loading of model checkpoints to a specified S3 bucket or locally (or both).

    :param project_name: The name of the project for organizing checkpoints
    :param bucket_name: The name of the S3 bucket to store checkpoints (Optional)
    :param save_locally: If True, saves checkpoints locally without deleting them (Optional)
    """
    def __init__(self, project_name: str, bucket_name=None, save_locally=False):
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.save_locally = save_locally
        self.client = None
        self.checkpoints = []

        if bucket_name:
            if os.getenv("AWS_SECRET_ACCESS_KEY") and os.getenv("AWS_ACCESS_KEY_ID"):
                self.client = boto3.client("s3")
                self.checkpoints = self.get_checkpoints()
            else:
                warnings.warn("AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID must be set to use S3 bucket.")

        checkpoint_dir = f"../{self.get_checkpoint_dir(self.project_name)}"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if not bucket_name and not save_locally:
            warnings.warn("Neither S3 bucket name provided nor save_locally is True. Checkpoints will not be persisted.")

    def get_checkpoints(self) -> List[Tuple[int, int]]:
        """
        Returns a list of tuples (epoch, batch_idx) of all checkpoints in the bucket or local directory.
        """
        if self.bucket_name and self.client:
            response = self.client.list_objects_v2(Bucket=self.bucket_name)
            if "Contents" in response:
                return sorted(
                    [self.decode_checkpoint_id(item["Key"]) for item in response["Contents"]]
                )
            warnings.warn("No files found in the bucket.")
        elif self.save_locally:
            checkpoint_files = glob.glob(f"{self.get_checkpoint_dir(self.project_name)}/*.pt")
            return sorted(
                [self.decode_checkpoint_id(os.path.basename(f)) for f in checkpoint_files]
            )
        else:
            warnings.warn("No checkpoints found locally.")
        return []
    
    # Names & ids

    @staticmethod
    def get_checkpoint_id(epoch: int, batch_idx: int) -> str:
        return f"checkpoint_epoch_{epoch}_batch_{batch_idx}"

    @classmethod
    def get_checkpoint_dir(cls, project_name: str) -> str:
        return f"checkpoints/{project_name}"

    @classmethod
    def get_checkpoint_path(cls, epoch: int, batch_idx: int, project_name: str) -> str:
        return f"{cls.get_checkpoint_dir(project_name)}/{cls.get_checkpoint_id(epoch, batch_idx)}.pt"

    @staticmethod
    def decode_checkpoint_id(checkpoint_id: str) -> Tuple[int, int]:
        parts = checkpoint_id.split("_")
        epoch = int(parts[-3])
        batch_idx = int(parts[-1].split(".")[0])
        return epoch, batch_idx

    # Data

    def _upload_file(self, file_path, object_name=None):
        if object_name is None:
            object_name = file_path
        self.client.upload_file(file_path, self.bucket_name, object_name)

    def save_checkpoint(
        self,
        checkpoint: Dict,
        epoch: int,
        batch_idx: int,
    ):
        file_path = self.get_checkpoint_path(epoch, batch_idx, self.project_name)
        rel_file_path = f"../{file_path}"
        torch.save(checkpoint, rel_file_path)

        if self.client:
            self._upload_file(rel_file_path, file_path)

        if not self.save_locally:
            os.remove(rel_file_path)

    def load_checkpoint(
        self, epoch: int, batch_idx: int,
    ) -> Dict:
        file_path = self.get_checkpoint_path(epoch, batch_idx, self.project_name)
        rel_file_path = f"../{file_path}"
        
        if self.client:
            object_name = os.path.basename(file_path)
            print(f"Downloading {object_name} from {self.bucket_name}...")
            self.client.download_file(self.bucket_name, object_name, rel_file_path)
        
        checkpoint = torch.load(rel_file_path)

        if not self.save_locally and self.bucket_name and self.client:
            os.remove(rel_file_path)

        return checkpoint
    
    def __iter__(self):
        for checkpoint in self.checkpoints:
            yield self.load_checkpoint(*checkpoint)

    def __len__(self):
        return len(self.checkpoints)

    def __getitem__(self, idx: Union[int, tuple]):
        if isinstance(idx, int):
            return self.load_checkpoint(*self.checkpoints[idx])

        elif isinstance(idx, tuple):
            if idx not in self.checkpoints:
                warnings.warn(f"Checkpoint {idx} not found in {self.bucket_name}.")

            return self.load_checkpoint(*idx)

        raise TypeError(f"Invalid argument type: {type(idx)}")

    def __contains__(self, checkpoint: Tuple[int, int]):
        return checkpoint in self.checkpoints

    def __repr__(self):
        return f"CheckpointManager({self.project_name}, {self.bucket_name})"
