import pandas as pd
import numpy as np
import pytest
from pathlib import Path

TEST_DATA = Path(__file__).parent / "data"

from gin import (
    RuntimeInflowBoundary,
    RuntimeDownstreamBoundary,
    RuntimeSedimentBoundary,
    InflowBoundaryConst,
    InflowBoundaryTS,
)


class TestRuntimeInflowBoundaryConst:
    @pytest.fixture
    def cib(self):
        ibs = [
            InflowBoundaryConst(ordinate=0.0, type="const", value=42.0),
            InflowBoundaryConst(ordinate=-5.0, type="const", value=12.0),
            InflowBoundaryConst(ordinate=15.0, type="const", value=15.0),
        ]
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2021-01-01")
        return RuntimeInflowBoundary(ibs, start, end)

    def test_const_returns_value_regardless_of_time(self, cib):
        np.testing.assert_array_equal(
            cib.get_flows(pd.Timestamp("2020-01-01")), [12, 54, 69]
        )

    def test_rechainage(self, cib):
        cib.rechainage((-3, -6, -2, 0, 10, 25))
        np.testing.assert_array_equal(
            cib.get_flows(pd.Timestamp("2020-01-01")), [0, 12, 12, 54, 54, 69]
        )

    @pytest.fixture
    def vib(self):
        ibs = [
            InflowBoundaryTS(
                ordinate=0.0, type="ts", value=TEST_DATA / "inflow_messy_bc_0.0.csv"
            ),
            InflowBoundaryConst(ordinate=-5.0, type="const", value=12.0),
            InflowBoundaryTS(
                ordinate=15.0, type="ts", value=TEST_DATA / "inflow_messy_bc_15.0.csv"
            ),
        ]
        start = pd.Timestamp("2020-01-01 10:0:0")
        end = pd.Timestamp("2020-01-02 3:0:0")
        return RuntimeInflowBoundary(ibs, start, end)

    def test_get_flows(self, vib):
        np.testing.assert_array_equal(
            vib.get_flows(pd.Timestamp("2020-01-01 23:51:00")),
            [12.0, 29.333, 50.433],
        )
        np.testing.assert_allclose(
            vib.get_flows(pd.Timestamp("2020-01-01 13:40:00")),
            [12, 15.39, 5.128843],
        )

    def test_messy_rechainge(self, vib):
        vib.rechainage((-5, -6, -2, 0, 3, 6, 25))
        np.testing.assert_allclose(
            vib.get_flows(pd.Timestamp("2020-01-01 13:40:00")),
            [0, 12, 12, 15.39, 15.39, 15.39, 5.128843],
        )

    @pytest.fixture
    def vib_simple(self):
        ibs = [
            InflowBoundaryTS(
                ordinate=2.0, type="ts", value=TEST_DATA / "inflow_simple_bc_2.0.csv"
            ),
            InflowBoundaryConst(ordinate=-5.0, type="const", value=12.0),
            InflowBoundaryTS(
                ordinate=7.0, type="ts", value=TEST_DATA / "inflow_simple_bc_7.0.csv"
            ),
        ]
        start = pd.Timestamp("2010-01-01 02:0:0")
        end = pd.Timestamp("2010-01-01 6:0:0")
        return RuntimeInflowBoundary(ibs, start, end)

    def test_get_flows_simple(self, vib_simple):
        np.testing.assert_array_equal(
            vib_simple.get_flows(pd.Timestamp("2010-01-01 4:0:0")),
            [12.0, 16, 30],
        )
        np.testing.assert_array_equal(
            vib_simple.get_flows(pd.Timestamp("2010-01-01 3:30:0")),
            [12.0, 15.5, 29],
        )

    def test_simple_rechainge(self, vib_simple):
        vib_simple.rechainage((-5, -6, -2, 0, 3, 6, 25))
        np.testing.assert_array_equal(
            vib_simple.get_flows(pd.Timestamp("2010-01-01 4:0:0")),
            [0, 12, 12, 12, 16, 16, 30],
        )
        np.testing.assert_array_equal(
            vib_simple.get_flows(pd.Timestamp("2010-01-01 1:30:0")),
            [0, 12, 12, 12, 13.5, 13.5, 25],
        )


"""
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

"""


class TestRuntimeDownstreamBoundary:
    def test_elevation_type(self):
        b = RuntimeDownstreamBoundary(type="elevation", value=5.0)
        result = b.value_at(pd.Timestamp("2020-01-01"))
        assert result == {"elevation": 5.0}

    def test_depth_type(self):
        b = RuntimeDownstreamBoundary(type="depth", value=1.5)
        result = b.value_at(pd.Timestamp("2020-01-01"))
        assert result == {"depth": 1.5}

    def test_normal_type(self):
        b = RuntimeDownstreamBoundary(type="normal", value=None)
        b.slope = 0.001
        b.wl_init = 2.0
        result = b.value_at(pd.Timestamp("2020-01-01"))
        assert result == {"normal": {"slope": 0.001, "wl_init": 2.0}}

    def test_ts_interpolation_returns_elevation_key(self):
        idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
        series = pd.Series([1.0, 3.0], index=idx)
        b = RuntimeDownstreamBoundary(type="elevation_timeseries", value=series)
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
