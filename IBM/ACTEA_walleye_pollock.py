# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2021, Trond Kristiansen, Niva
# Jan 2021 Simplified by Knut-Frode Dagestad, MET Norway, and adapted to Kvile et al. (2018)

# Modified for Walleye pollock, 2025 by Trond Kristiansen

from opendrift.models.oceandrift import Lagrangian3DArray, OceanDrift
import logging
import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class LarvalFishElement(Lagrangian3DArray):
    """
    Extending Lagrangian3DArray with specific properties for larval and juvenile stages of fish
    """

    variables = Lagrangian3DArray.add_variables(
        [
            (
                "diameter",
                {"dtype": np.float32, "units": "m", "default": 0.0014},
            ),  # for Walleye pollock
            (
                "neutral_buoyancy_salinity",
                {"dtype": np.float32, "units": "PSU", "default": 31.25},
            ),  # for NEA Cod
            (
                "stage_fraction",
                {
                    "dtype": np.float32,  # to track percentage of development time completed
                    "units": "",
                    "default": 0.0,
                },

            ),
            (
                "age",
                {
                    "dtype": np.float32,  # to track age of each egg
                    "units": "days",
                    "default": 0.0,
                },

            ),
            (
                "hatched",
                {
                    "dtype": np.uint8,  # 0 for eggs, 1 for larvae
                    "units": "",
                    "default": 0,
                },
            ),
            ("length", {"dtype": np.float32, "units": "mm", "default": 0}),
            ("weight", {"dtype": np.float32, "units": "g", "default": 0.38}),
            ("density", {"dtype": np.float32, "units": "g/kg", "default": 1024}),
            (
                "survival",
                {"dtype": np.float32, "units": "", "default": 1.0},  # Not yet used
            ),
        ]
    )


class LarvalFish(OceanDrift):
    ElementType = LarvalFishElement

    required_variables = {
        "x_sea_water_velocity": {"fallback": 0},
        "y_sea_water_velocity": {"fallback": 0},
        "sea_surface_wave_stokes_drift_x_velocity": {"fallback": 0},
        "sea_surface_wave_stokes_drift_y_velocity": {"fallback": 0},
        "sea_surface_wave_significant_height": {"fallback": 0},
        "sea_surface_height": {"fallback": 0},
        "x_wind": {"fallback": 0},
        "y_wind": {"fallback": 0},
        "land_binary_mask": {"fallback": None},
        "sea_floor_depth_below_sea_level": {"fallback": 100},
        "ocean_vertical_diffusivity": {"fallback": 0.01, "profiles": True},
        "ocean_mixed_layer_thickness": {"fallback": 50},
        "sea_water_temperature": {"fallback": 10, "profiles": True},
        "sea_water_salinity": {"fallback": 34, "profiles": True},
    }

    def __init__(self, *args, **kwargs):

        # Calling general constructor of parent class
        super(LarvalFish, self).__init__(*args, **kwargs)

        # IBM configuration options
        self._add_config(
            {
                "IBM:fraction_of_swimming_horisontal": {
                    "type": "float",
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "units": "fraction",
                    "description": "Fraction of horisontal swimming speed vertically",
                    "level": 20,
                },
            }
        )
        self.constant_egg_density = False
        self.dark_or_light_egg_density = "light"
        self.port_townsend_eggs = True
        self._set_config_default("drift:vertical_mixing", True)

    def update_terminal_velocity(self, Tprofiles=None,
                                 Sprofiles=None, z_index=None):
        """Calculate terminal velocity for Pelagic Egg

        according to
        S. Sundby (1983): A one-dimensional model for the vertical
        distribution of pelagic fish eggs in the mixed layer
        Deep Sea Research (30) pp. 645-661

        Method copied from ibm.f90 module of LADIM:
        Vikebo, F., S. Sundby, B. Aadlandsvik and O. Otteraa (2007),
        Fish. Oceanogr. (16) pp. 216-228
        """
        g = 9.81  # ms-2
        eggsize = self.elements.diameter  # 0.0014 for NEA Cod
        density_egg = self.elements.density

        # prepare interpolation of temp, salt
        if not (Tprofiles is None and Sprofiles is None):
            if z_index is None:
                z_i = range(Tprofiles.shape[0])  # evtl. move out of loop
                # evtl. move out of loop
                z_index = interp1d(-self.environment_profiles['z'],
                                   z_i, bounds_error=False)
            zi = z_index(-self.elements.z)
            upper = np.maximum(np.floor(zi).astype(np.uint8), 0)
            lower = np.minimum(upper + 1, Tprofiles.shape[0] - 1)
            weight_upper = 1 - (zi - upper)

        # do interpolation of temp, salt if profiles were passed into
        # this function, if not, use reader by calling self.environment
        if Tprofiles is None:
            T0 = self.environment.sea_water_temperature
        else:
            T0 = Tprofiles[upper, range(Tprofiles.shape[1])] * \
                 weight_upper + \
                 Tprofiles[lower, range(Tprofiles.shape[1])] * \
                 (1 - weight_upper)
        if Sprofiles is None:
            S0 = self.environment.sea_water_salinity
        else:
            S0 = Sprofiles[upper, range(Sprofiles.shape[1])] * \
                 weight_upper + \
                 Sprofiles[lower, range(Sprofiles.shape[1])] * \
                 (1 - weight_upper)

        # The density difference between a pelagic egg and the ambient water
        # is regulated by their salinity difference through the
        # equation of state for sea water.
        # The Egg has the same temperature as the ambient water and its
        # salinity is regulated by osmosis through the egg shell.
        DENSw = self.sea_water_density(T=T0, S=S0)
        dr = DENSw - density_egg  # density difference

        # water viscosity
        my_w = 0.001 * (1.7915 - 0.0538 * T0 + 0.007 * (T0 ** (2.0)) - 0.0023 * S0)
        # ~0.0014 kg m-1 s-1

        # terminal velocity for low Reynolds numbers
        W = (1.0 / my_w) * (1.0 / 18.0) * g * eggsize ** 2 * dr

        # check if we are in a Reynolds regime where Re > 0.5
        highRe = np.where(W * 1000 * eggsize / my_w > 0.5)

        # Use empirical equations for terminal velocity in
        # high Reynolds numbers.
        # Empirical equations have length units in cm!
        my_w = 0.01854 * np.exp(-0.02783 * T0)  # in cm2/s
        d0 = (eggsize * 100) - 0.4 * \
             (9.0 * my_w ** 2 / (100 * g) * DENSw / dr) ** (1.0 / 3.0)  # cm
        W2 = 19.0 * d0 * (0.001 * dr) ** (2.0 / 3.0) * (my_w * 0.001 * DENSw) ** (-1.0 / 3.0)
        # cm/s
        W2 = W2 / 100.  # back to m/s

        W[highRe] = W2[highRe]
        self.elements.terminal_velocity = W

    def update_egg_density(self):
        """
        Update the egg density for the given stage fractions. This method is based on 
        laboratory work by Dr. Ben Laurel and requires `sea_water_temperature` and 
        `sea_water_salinity` to be available in the model to calculate the 
        water density. The water density is used when calculating the water density and
        the terminal velocity of the eggs.

        Parameters:
        - self: The current instance of the class.

        Returns:
        Updates self.elements.density

        """
        eggs = np.where(self.elements.hatched == 0)[0]
        if len(eggs) > 0:
            """
            Define the arrays based on Ben Laurels laboratory work (2024). The data used are 
            for Walleye Pollock exposed to light and the impacts development has on egg stage.
            """

            if self.port_townsend_eggs:
                if self.dark_or_light_egg_density == "dark":
                    egg_density = np.array([1.024446663, 1.024665364, 1.024937853,
                    1.02484257, 1.024994585, 1.024975082, 1.024875345, 1.024765756,
                    1.024764591, 1.024751739, 1.024760701, 1.024879361, 1.025103333, 
                    1.025475083, 1.026092269, 1.026719133]) * 1000.

                elif self.dark_or_light_egg_density == "light":
                    egg_density = np.array([1.024413091, 1.02464518, 1.024770682,
                    1.025047026, 1.024993654, 1.025047998, 1.024956875, 1.024893251,
                    1.024938948, 1.024999149, 1.025016702, 1.025019958, 1.025217969,
                    1.025624979, 1.025998044, 1.026675837]) * 1000.

            else: 
                if self.dark_or_light_egg_density == "dark":
                    egg_density = np.array([1.024965885, 1.025167579, 1.025282136, 1.0253613,
                                            1.025440456, 1.025406614, 1.02533438, 1.025357889,
                                            1.025349958, 1.025311835, 1.025300872, 1.025253598,
                                            1.025457323, 1.025680634, 1.026287275, 1.026824609]) * 1000.

                elif self.dark_or_light_egg_density == "light":
                    egg_density = np.array([1.024966891, 1.025198241, 1.025357486, 1.025462898,
                                            1.025572679, 1.025602902, 1.025576521, 1.025518015,
                                            1.025496942, 1.025441822, 1.025372264, 1.02525809,
                                            1.02542082, 1.025739589, 1.026090628, 1.026637115]) * 1000.

            egg_stage = np.array(
                [0, 0.066666667, 0.133333333, 0.2, 0.266666667, 0.333333333, 0.4, 0.466666667, 0.533333333,
                 0.6, 0.666666667, 0.733333333, 0.8, 0.866666667, 0.933333333, 1])

            # Function to find the index of the nearest value
            def find_nearest(array, value):
                idx = (np.abs(array - value)).argmin()
                return idx

            # Find the corresponding egg_density values
            density = np.array(
                [egg_density[find_nearest(egg_stage, value)] for value in self.elements.stage_fraction[eggs]])
            self.elements.density[eggs] = density

    def update_fish_eggs(self):
        """

        Updates the fish eggs in the simulation.

        This method calculates the development progress of fish eggs and updates their attributes accordingly.

        Parameters:
            self (object): The instance of the class.

        Returns:
            Updates:    
                `self.elements.hatched` and `self.elements.stage_fraction` attributes.

        """
        # Hatching of eggs
        eggs = np.where(self.elements.hatched == 0)[0]
        if len(eggs) > 0:
            # Equation from Ben Laurel (May 14th 2024 email)
            amb_duration = 0.0506982 + 0.0149187 * self.environment.sea_water_temperature[eggs]

            # Total egg development time (days) according to ambient temperature
            days_in_timestep = self.time_step.total_seconds() / (
                    60 * 60 * 24
            )  # The fraction of a day completed in one time step
            amb_fraction = (
                    days_in_timestep * amb_duration
            )  # Fraction of development time completed during present time step

            # Add fraction completed during present timestep to cumulative fraction completed
            self.elements.stage_fraction[eggs] += amb_fraction
            self.elements.age[eggs] += days_in_timestep
            hatching = np.where(self.elements.stage_fraction[eggs] >= 1)[0]

            if len(hatching) > 0:
                logger.debug("Hatching %s eggs" % len(hatching))
                self.elements.hatched[eggs[hatching]] = (
                    1  # Eggs with total development time completed are hatched (1)
                )

    def update(self):

        self.stokes_drift()
        self.update_fish_eggs()

        # To compare the effect of constant egg density with one that varies under
        # development we run two different scenarios using "self.constant_egg_density" 
        # option on or off. This is set in ACTEA_run_drift.eggs.py
        if not self.constant_egg_density:
            self.update_egg_density()

        self.advect_ocean_current()
        self.vertical_mixing()
        self.update_terminal_velocity()
