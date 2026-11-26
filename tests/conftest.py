"""Keep extension tests distinct from public API fallback tests."""

from importlib import import_module

import pytest


def pytest_runtest_setup(item):
    if item.get_closest_marker("cuda_extension"):
        try:
            import_module("rcabeam._cuda_impl")
        except ModuleNotFoundError as error:
            if error.name != "rcabeam._cuda_impl":
                raise
            pytest.skip("requires the built rcabeam CUDA extension")
