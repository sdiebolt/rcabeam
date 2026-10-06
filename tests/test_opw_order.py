"""Pure Cartesian traversal checks for OPW's spatial tiling."""

import numpy as np
import pytest

from rcabeam.ensemble import opw_voxel_order


def test_opw_order_preserves_grid_and_tails() -> None:
    """Patch ordering is a permutation and scatters back to canonical indices."""
    for shape in ((4, 3, 8), (3, 2, 5), (1, 2, 1), (2, 1, 4), (5, 3, 2)):
        order = opw_voxel_order(shape)
        expected = np.arange(np.prod(shape))
        np.testing.assert_array_equal(np.sort(order), expected)
        restored = np.empty_like(order)
        restored[order] = expected[order]
        np.testing.assert_array_equal(restored, expected)
    np.testing.assert_array_equal(opw_voxel_order((4, 3, 8))[:8], [0, 1, 2, 3, 24, 25, 26, 27])
    with pytest.raises(ValueError):
        opw_voxel_order((0, 2, 4))
