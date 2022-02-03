import pytest

# The following code allows custom flags when running pytest. In the future we could add --integrationtest and --unittest etc.
def pytest_addoption(parser):
    parser.addoption(
        "--buildtest", action="store_true", default=False, help="run build tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "buildtest: add --buildtest to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--buildtest"):
        # --buildtest given in cli: do not skip build tests
        return
    skip_slow = pytest.mark.skip(reason="need --buildtest option to run")
    for item in items:
        if "buildtest" in item.keywords:
            item.add_marker(skip_slow)