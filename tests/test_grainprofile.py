"""
Unit tests for grainprofile.py.

NOTE: as of this writing, test_get_grain_props_matches_docstring_example
FAILS against the current implementation. That's intentional -- it encodes
the worked example from get_grain_props' own docstring as the expected
(correct) behaviour, and the current code does not produce it. See the
review notes for the two bugs in get_grain_props (wrong column selected
per-profile, and a spurious extra element in `prop`).
"""

import numpy as np
import pytest

from grainprofile import get_representative_grain_sizes, get_grain_props


class TestGetRepresentativeGrainSizes:
    def test_two_bins(self):
        # boundaries b0=1, b1=2, b2=4 (column 0 of each row; other columns
        # are per-profile data and irrelevant to this function)
        cfds = [
            [1.0, 0, 0],
            [2.0, 10, 5],
            [4.0, 100, 50],
        ]
        result = get_representative_grain_sizes(cfds)
        expected = np.array(
            [
                2 ** ((np.log(1.0) + np.log(2.0)) / 2),
                2 ** ((np.log(2.0) + np.log(4.0)) / 2),
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_returns_nbins_values_not_nbins_plus_one(self):
        # 3 bin boundaries -> 2 bins -> 2 representative sizes
        cfds = [[1.0, 0], [2.0, 10], [4.0, 100]]
        result = get_representative_grain_sizes(cfds)
        assert len(result) == 2


class TestGetGrainProps:
    def test_matches_docstring_example(self):
        """From get_grain_props' own docstring:

        2 bins, 3 lith groups. Profile 0's cumulative-frequency column is
        [0, 10, 100], giving proportions 0.1 (bin 1) and 0.9 (bin 2).
        The lith table (bin-major, 2 bins x 3 liths) is:
            b1 l1 100   b1 l2   0   b1 l3   0
            b2 l1  50   b2 l2   0   b2 l3  50
        Expected result:
            [[0.1,  0,    0   ],
             [0.45, 0,    0.45]]
        """
        # column 0 = bin boundaries (unused by this function directly other
        # than to size things), column 1 = profile 0's cumulative freq
        cfds = [
            [1.0, 0],
            [2.0, 10],
            [4.0, 100],
        ]
        lithtab = [
            [100],  # bin1, lith1
            [0],  # bin1, lith2
            [0],  # bin1, lith3
            [50],  # bin2, lith1
            [0],  # bin2, lith2
            [50],  # bin2, lith3
        ]

        result = get_grain_props(0, cfds, lithtab)

        expected = np.array(
            [
                [0.1, 0.0, 0.0],
                [0.45, 0.0, 0.45],
            ]
        )
        assert result.shape == (2, 3), f"expected (2, 3), got {result.shape}"
        np.testing.assert_allclose(result, expected)

    def test_rows_sum_to_bin_proportions(self):
        """Regardless of lith split, each row of the result should sum to
        the bin's overall proportion of the profile (0.1 and 0.9 here)."""
        cfds = [[1.0, 0], [2.0, 10], [4.0, 100]]
        lithtab = [[100], [0], [0], [50], [0], [50]]

        result = get_grain_props(0, cfds, lithtab)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, [0.1, 0.9])

    def test_selects_correct_profile_column(self):
        """Two profiles in the same cfds table should give different,
        independent results depending on which pid is requested."""
        cfds = [
            [1.0, 0, 0],
            [2.0, 10, 90],
            [4.0, 100, 100],
        ]
        # single lith group so the lithtab reshape is trivial
        lithtab = [[1], [1]]

        prop0 = get_grain_props(0, cfds, lithtab).ravel()
        prop1 = get_grain_props(1, cfds, lithtab).ravel()

        np.testing.assert_allclose(prop0, [0.1, 0.9])
        np.testing.assert_allclose(prop1, [0.9, 0.1])

    def test_single_lith_group_shape(self):
        cfds = [[1.0, 0], [2.0, 10], [4.0, 100]]
        lithtab = [[1], [1]]  # nlith == 1
        result = get_grain_props(0, cfds, lithtab)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.ravel(), [0.1, 0.9])
