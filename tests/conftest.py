import pytest


@pytest.fixture
def is_snapshot_update(request):
    return request.config.getoption("--snapshot-update")
