"""
Unit tests for the Runtime*Boundary classes in gin.py (time-series lookup /
interpolation helpers).

IMPORTANT: these tests import gin.py directly, which currently fails at
import time (see bug #1 in the review notes: `processed_sediment_boundary`
must be `_processed_sediment_boundary` for pydantic to accept it as a
PrivateAttr). Apply that one-line rename before running this file.
"""

import pandas as pd
import pytest

from gin import (
    RuntimeInflowBoundary,
    RuntimeDownstreamBoundary,
    RuntimeSedimentBoundary,
)


class TestRuntimeInflowBoundaryConst:
    def test_const_returns_value_regardless_of_time(self):
        b = RuntimeInflowBoundary(ordinate=0.0, type="const", value=42.0)
        assert b.value_at(pd.Timestamp("2020-01-01")) == 42.0
        assert b.value_at(pd.Timestamp("2030-01-01")) == 42.0


class TestRuntimeInflowBoundaryTS:
    @pytest.fixture
    def series(self):
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        return pd.Series([10.0, 20.0, 40.0], index=idx)

    def test_exact_timestamp_hit(self, series):
        b = RuntimeInflowBoundary(ordinate=0.0, type="ts", value=series)
        assert b.value_at(pd.Timestamp("2020-01-02")) == 20.0

    def test_interpolates_between_points(self, series):
        b = RuntimeInflowBoundary(ordinate=0.0, type="ts", value=series)
        # halfway between day 1 (10) and day 2 (20) -> 15
        result = b.value_at(pd.Timestamp("2020-01-01 12:00"))
        assert result == pytest.approx(15.0)

    def test_clamps_before_start(self, series):
        b = RuntimeInflowBoundary(ordinate=0.0, type="ts", value=series)
        assert b.value_at(pd.Timestamp("2019-01-01")) == 10.0

    def test_clamps_after_end(self, series):
        b = RuntimeInflowBoundary(ordinate=0.0, type="ts", value=series)
        assert b.value_at(pd.Timestamp("2021-01-01")) == 40.0


class TestRuntimeDownstreamBoundary:
    def test_elevation_type(self):
        b = RuntimeDownstreamBoundary(ordinate=0.0, type="elevation", value=5.0)
        result = b.value_at(pd.Timestamp("2020-01-01"))
        assert result == {"elevation": 5.0}

    def test_depth_type(self):
        b = RuntimeDownstreamBoundary(ordinate=0.0, type="depth", value=1.5)
        result = b.value_at(pd.Timestamp("2020-01-01"))
        assert result == {"depth": 1.5}

    def test_normal_type(self):
        b = RuntimeDownstreamBoundary(ordinate=0.0, type="normal", value=None)
        b.slope = 0.001
        b.hinit = 2.0
        result = b.value_at(pd.Timestamp("2020-01-01"))
        assert result == {"normal": {"slope": 0.001, "hinit": 2.0}}

    def test_ts_interpolation_returns_elevation_key(self):
        idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
        series = pd.Series([1.0, 3.0], index=idx)
        b = RuntimeDownstreamBoundary(
            ordinate=0.0, type="elevation_timeseries", value=series
        )
        result = b.value_at(pd.Timestamp("2020-01-01 12:00"))
        assert result == {"elevation": pytest.approx(2.0)}


class TestRuntimeSedimentBoundary:
    def test_unravel_reshapes_to_nbins_by_nlith(self):
        import numpy as np

        b = RuntimeSedimentBoundary(
            ordinate=0.0, type="const", nbins=2, nlith=3, value=None
        )
        flat = [1, 2, 3, 4, 5, 6]
        result = b.unravel(flat)
        np.testing.assert_array_equal(result, np.array([[1, 2, 3], [4, 5, 6]]))

    def test_const_value_at_returns_stored_array(self):
        import numpy as np

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = RuntimeSedimentBoundary(
            ordinate=0.0, type="const", nbins=2, nlith=2, value=arr
        )
        np.testing.assert_array_equal(b.value_at(pd.Timestamp("2020-01-01")), arr)

    def test_ts_value_at_interpolates_and_unravels(self):
        import numpy as np

        idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
        # each row is a flattened nbins*nlith vector
        df = pd.DataFrame([[0.0, 0.0, 0.0, 0.0], [4.0, 8.0, 12.0, 16.0]], index=idx)
        b = RuntimeSedimentBoundary(ordinate=0.0, type="ts", nbins=2, nlith=2, value=df)
        result = b.value_at(pd.Timestamp("2020-01-01 12:00"))
        expected = np.array([[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose(result, expected)
