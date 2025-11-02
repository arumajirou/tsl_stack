\
import os, pytest

@pytest.fixture(scope="session", autouse=True)
def _seed():
    os.environ.setdefault("TSL_SEED","2077")
    yield
