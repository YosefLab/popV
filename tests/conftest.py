import shutil
import pytest

def pytest_addoption(parser):
    """Docstring for pytest_addoption."""
    parser.addoption(
        "--accelerator",
        action="store",
        default="cpu",
        help="Option to specify which accelerator to use for tests.",
    )
    parser.addoption(
        "--devices",
        action="store",
        default="auto",
        help="Option to specify which devices to use for tests.",
    )

@pytest.fixture(scope="session")
def accelerator(request):
    """Docstring for accelerator."""
    return request.config.getoption("--accelerator")


@pytest.fixture(scope="session")
def devices(request):
    """Docstring for devices."""
    return request.config.getoption("--devices")
