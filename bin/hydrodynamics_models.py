import math
import numpy as np
import pandas as pd
from functools import lru_cache

import utils
from gin import GrateConfig
from channel import Channel, Loc


class HydroDynamicModel:
    def __init__(self, cfg: GrateConfig, chan: Channel):
        """
        A design choice is that all the getters on this class should return
        info from stored arrays, not go get the info from elsewhere.

        Functions that start with _calc actually calculate values
        """

        self._cfg = cfg
        self._channel = chan
        self._cs = self._channel._chainpts()

        # make sure the inflow and sediment boundary conditions know about (potentially) new chainage
        cfg._processed_inflow.rechainage(self._cs)
        cfg._processed_sediment_boundary.rechainage(self._cs)

        # store these variables for current timestep
        self._Sf = np.empty(len(self._cs))
        self._vel = np.empty(len(self._cs))
        self._flow = np.empty(len(self._cs))
        self._wl = np.empty(len(self._cs))

        self.initialize(self._cfg.simulation_time.start)

    def initialize(self, t: pd.Timestamp):
        """Set water level (elevation above same datum cross-section profile y
        uses) to initial value.
        """
        # use the downstream for water level. 'normal' is a special case,
        # use wl_init
        tv = self._cfg._processed_downstream_boundary.value_at(t)
        # this is elevation
        if "normal" in tv:
            v = tv["normal"]["wl_init"]
        else:
            v = self._calc_ds_wl(t)

        # set cross-section/layer caches except for gsv which we must miss
        # since not sensible hydraulics yet
        self._channel.set_layer_arrays(self, do_gsv=False)

        # initialize water level to downstream value shifted by bed
        self._wl[-1] = v
        for c in range(len(self._cs) - 2, -1, -1):
            self._wl[c] = (
                self._wl[c + 1]
                - self._channel.get_min_bed_level()[c + 1]
                + self._channel.get_min_bed_level()[c]
            )

        # starting flow
        self._flow = self._cfg._processed_inflow.get_flows(t)

        # run the first step, this gets our Sf and vel set which allows
        # channel to initialize its variables
        self.update_water_level(t)
        self._channel.set_layer_arrays(self)

    def get_beta(self):
        """Momentum correction factor"""
        return self._cfg.morphological.beta

    def get_vel(self):
        """Previously calculated u or velocity"""
        return self._vel

    def get_flow(self):
        """Previously calculated flow"""
        return self._flow

    def get_Sf(self):
        """Previously calculated Sf"""
        return self._Sf

    def get_wl(self):
        return self._wl

    def _calc_vel(self, c: int, wl: float):
        """The mean velocity, ie Q/A at current step"""
        return self._flow[c] / self._channel.calc_area(c, wl, Loc.ALL)

    def _calc_Sf(self, c: int, wl: float):
        """Return friction slope, ie. Q abs(Q) / K^2"""
        Q = self._flow[c]
        # print(
        #     f"calc_Sf at {c}, min_bed_level is {self._channel.get_min_bed_level()[c]}"
        # )
        return Q * abs(Q) / self._channel.calc_conveyance(c, wl) ** 2

    def _calc_ds_wl(self, t: pd.Timestamp):
        """Return downstream water level of water."""
        lastchain = len(self._cs) - 1
        tv = self._cfg._processed_downstream_boundary.value_at(t)
        if "elevation" in tv:
            return tv["elevation"]
        if "depth" in tv:
            return tv["depth"] + self._channel.get_min_bed_level()[lastchain]
        if "normal" in tv:
            slope = tv["normal"]["slope"]
            Q = self._flow[lastchain]

            def f(wl):
                K = self._channel._calc_conveyance(lastchain, wl)
                return Q - K * math.sqrt(slope)

            try:
                wl = utils.find_root(
                    f, self._wl[lastchain], self._channel.get_min_bed_level()[lastchain]
                )
            except Exception as exp:
                raise RuntimeError("get_ds_wl failed to find root") from exp

            return wl

        raise ValueError(f"Unknown downstream boundary type: {tv['type']}")

    def update_water_level(self):
        raise NotImplementedError()


class QuasiSteadyModel(HydroDynamicModel):
    def conservation_of_energy(self, c: int, wl: float):
        """Calculate equation 5.7, the change in energy between me and
        downstream

        f(h_i^*) =
            h_(i) + (β_i (u_i^* )^2)/2g
            - (h_(i+1) + (β_(i+1) u_(i+1)^2)/2g)
            + (S_0-S_f^*)Δx

        h_(i) is the water level (elevation above datum)
        S_0 is ZERO since using water level
        h_0 is water level at most upstream
        u is velocity of water.
        i is chain point index (c in this case)

        We should have already sorted at c+1
        """

        dc = self._cs[c + 1] - self._cs[c]
        sf = (self._calc_Sf(c, wl) + self._Sf[c + 1]) / 2
        g = 9.8
        return (
            wl
            + self.get_beta()
            * (self._calc_vel(c, wl) ** 2 - self._vel[c + 1] ** 2)
            / (2 * g)
            - self._wl[c + 1]
            - sf * dc
        )

    def update_water_level(self, t: pd.Timestamp):
        """

        Parameters
        ----------
        t: pd.Timestamp
            Timestep
        """

        # get the most downstream water level
        self._wl[-1] = self._calc_ds_wl(t)

        # current flow
        self._flow = self._cfg._processed_inflow.get_flows(t)

        # use wl[i+1] to calculate wl[i]
        for i in range(len(self._cs) - 2, -1, -1):
            self._Sf[i + 1] = self._calc_Sf(i + 1, self._wl[i + 1])
            self._vel[i + 1] = self._calc_vel(i + 1, self._wl[i + 1])

            @lru_cache(maxsize=1000)
            def f(wl):
                return self.conservation_of_energy(i, wl)

            init_wl = (
                self._wl[i + 1]
                - self._channel.get_min_bed_level()[i + 1]
                + self._channel.get_min_bed_level()[i]
            )

            try:
                wl = utils.find_root(
                    f,
                    x0=init_wl,
                    lower=self._channel.get_min_bed_level()[i],
                    max_growth=100.0,
                )

            except Exception as exp:
                wls = np.arange(
                    self._channel.get_min_bed_level()[i] + 0.0001,
                    self._channel.get_min_bed_level()[i] + 10.0,
                    0.0001,
                )
                f = np.array([f(wl) for wl in wls])
                Sfs = np.array([self._calc_Sf(i, wl) for wl in wls])
                us = np.array([self._calc_vel(i, wl) for wl in wls])
                df = pd.DataFrame({"wl": wls, "f": f, "Sf": Sfs, "u": us})
                df = df[(-1 < df.f) & (df.f < 1)]
                df.to_csv("root_finding_failure_f_values.csv", index=False)
                self._channel._xss[i].df.to_csv("failured_profile.csv", index=False)
                raise RuntimeError(
                    f"find_root failed at cross-section {i} chainage={self._cs[i]}, f saved in root_finding_failure_f_values.csv"
                ) from exp

            self._wl[i] = wl
            self._Sf[i] = self._calc_Sf(i, wl)
            self._vel[i] = self._calc_vel(i, wl)

            # print(f"Finished {t=} {i=} {wl=}")

        # tell channel that wl is fixed for this iteration so it an store its
        # variables that depend on wl
        self._channel.set_profile_arrays(self._wl)


class DynamicWaveModel(HydroDynamicModel):
    def initialize(self, t: pd.Timestamp):
        raise NotImplementedError(f"DynamicWaveModel is unusable at time {t}")
