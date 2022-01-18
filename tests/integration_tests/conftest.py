from pathlib import Path
from shutil import copytree

import pytest


@pytest.fixture
def test_data_path(tmp_path):
    """ Return pathlib.Path pointing to temporary copy of tests/example_data

   Use a fresh temporary directory for each test.
   """
    master_path = (Path(__file__) / '../../test_data').resolve()
    data_path = Path(tmp_path) / 'test_data'

    assert master_path.exists()
    assert not data_path.exists()
    copytree(str(master_path), str(data_path))
    return data_path
