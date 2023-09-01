import os
import shutil

import boto3
import pytest
import torch
from moto import mock_s3

from devinterp.ops.storage import (
    CompositeStorageProvider,
    LocalStorageProvider,
    S3StorageProvider,
)


@pytest.fixture(scope="function")
def aws_env_setup():
    remove_aws_key_env = False
    remove_aws_secret_env = False

    if not os.getenv("AWS_ACCESS_KEY_ID"):
        remove_aws_key_env = True
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"

    if not os.getenv("AWS_SECRET_ACCESS_KEY"):
        remove_aws_secret_env = True
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"

    yield

    if remove_aws_key_env:
        del os.environ["AWS_ACCESS_KEY_ID"]

    if remove_aws_secret_env:
        del os.environ["AWS_SECRET_ACCESS_KEY"]


def test_local_storage():
    storage = LocalStorageProvider(root_dir="test_data")
    tensor = torch.Tensor([1, 2, 3])
    storage.save_file("file1", tensor)

    loaded_tensor = storage.load_file("file1")
    try:
        assert torch.equal(tensor, loaded_tensor)
        assert "file1" in storage.get_file_ids()
    finally:
        shutil.rmtree("test_data")


@mock_s3
def test_s3_storage(aws_env_setup):
    # Setup
    bucket_name = "test-bucket"
    conn = boto3.client("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    storage = S3StorageProvider(bucket_name=bucket_name)
    tensor = torch.Tensor([1, 2, 3])

    storage.save_file("file1", tensor)

    loaded_tensor = storage.load_file("file1")
    assert torch.equal(tensor, loaded_tensor)
    assert "file1" in storage.get_file_ids()


@mock_s3
def test_composite_storage(aws_env_setup):
    # Setup
    bucket_name = "test-bucket"
    conn = boto3.client("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket_name)

    local_storage = LocalStorageProvider(root_dir="test_data")
    s3_storage = S3StorageProvider(
        bucket_name="test-bucket"
    )  # Assuming you've set this up

    storage = CompositeStorageProvider([local_storage, s3_storage])

    tensor = torch.Tensor([1, 2, 3])
    storage.save_file("file1", tensor)

    try:
        loaded_tensor = storage.load_file("file1")
        assert torch.equal(tensor, loaded_tensor)

        for _storage in storage.providers:
            ids = _storage.get_file_ids()
            assert "file1" in ids
    finally:
        shutil.rmtree("test_data")


# Add teardown code to cleanup files, buckets etc.
