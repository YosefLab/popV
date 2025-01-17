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
        "--cuml",
        action="store",
        default=False,
        help="Option to specify whether cuml is used.",
    )


@pytest.fixture(scope="session")
def accelerator(request):
    """Docstring for accelerator."""
    return request.config.getoption("--accelerator")


@pytest.fixture(scope="session")
def devices(request):
    """Docstring for cuml."""
    return request.config.getoption("--cuml")
