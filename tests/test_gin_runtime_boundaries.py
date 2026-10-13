import pandas as pd
import numpy as np
import pytest
from pathlib import Path

TEST_DATA = Path(__file__).parent / "data"

from gin import (
    RuntimeInflowBoundary,
    RuntimeDownstreamBoundary,
    RuntimeSedimentBoundaries,
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


class TestRuntimeSedimentBoundaries:
    @pytest.fixture
    def ordinates(self):
        return [4.0, 7.0, 9.0]

    @pytest.fixture
    def times(self):
        return pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 02:00"])

    @pytest.fixture
    def make_boundaries(self, ordinates):
        """Factory fixture creating constant or time-dependent boundaries."""

        def _make(times=None):
            if times is None:
                # Shape: (nboundaries, nbins, nlith)
                values = np.array([10.0, 20.0, 30.0]).reshape(3, 1, 1)
            else:
                # Shape: (ntimes, nboundaries, nbins, nlith)
                values = np.array(
                    [
                        [10.0, 20.0, 30.0],
                        [20.0, 40.0, 60.0],
                    ]
                ).reshape(2, 3, 1, 1)

            return RuntimeSedimentBoundaries(ordinates, times, values)

        return _make

    @pytest.fixture
    def constant_boundaries(self, make_boundaries):
        return make_boundaries()

    @pytest.fixture
    def time_boundaries(self, make_boundaries, times):
        return make_boundaries(times)

    def get_inputs(self, bc, t=None):
        """Extract the scalar input at each chainage for easy assertions."""
        values = bc.value_at(t)
        return values[:, 0, 0]

    def test_constant_boundaries(self, constant_boundaries):
        np.testing.assert_array_equal(
            self.get_inputs(constant_boundaries),
            [10.0, 20.0, 30.0],
        )

    def test_rechainage_exact_matches(self, constant_boundaries):
        bc = constant_boundaries
        bc.rechainage([3, 4, 5, 6, 7, 8, 9, 10, 11])

        np.testing.assert_array_equal(
            self.get_inputs(bc),
            [0, 10, 0, 0, 20, 0, 30, 0, 0],
        )

    def test_rechainage_moves_inputs_downstream(self, constant_boundaries):
        bc = constant_boundaries
        bc.rechainage([3, 3.9, 5, 6, 7, 8.9, 9.2])

        np.testing.assert_array_equal(
            self.get_inputs(bc),
            [0, 0, 10, 0, 20, 0, 30],
        )

    def test_rechainage_sums_inputs_at_same_chainage(self, constant_boundaries):
        bc = constant_boundaries
        bc.rechainage([3, 7, 10])

        np.testing.assert_array_equal(
            self.get_inputs(bc),
            [0, 30, 30],
        )

    def test_rechainage_drops_inputs_beyond_last_chainage(self, constant_boundaries):
        bc = constant_boundaries
        bc.rechainage([3, 5, 8])

        np.testing.assert_array_equal(
            self.get_inputs(bc),
            [0, 10, 20],
        )

    def test_rechainage_with_time_dependent_values(self, time_boundaries, times):
        bc = time_boundaries
        bc.rechainage([3, 7, 10])

        np.testing.assert_array_equal(
            self.get_inputs(bc, times[0]),
            [0, 30, 30],
        )

        np.testing.assert_array_equal(
            self.get_inputs(bc, times[1]),
            [0, 60, 60],
        )

    def test_time_interpolation_after_rechainage(self, time_boundaries):
        bc = time_boundaries
        bc.rechainage([3, 7, 10])

        t = pd.Timestamp("2020-01-01 01:00")

        np.testing.assert_allclose(
            self.get_inputs(bc, t),
            [0, 45, 45],
        )
