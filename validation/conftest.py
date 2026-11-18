import os

os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "numerical: Network solver benchmark (slow, about 1 GB RAM)")
