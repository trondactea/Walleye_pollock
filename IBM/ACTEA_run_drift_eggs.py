import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import ACTEA_common
import gcsfs
import numpy as np
import openpyxl
import pandas as pd
from ACTEA_config_drift import ActeaConf
from dateutil.relativedelta import relativedelta
from google.cloud import storage

import opendrift
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers.reader_netCDF_CF_generic import Reader

sys.path.append("../../ACTEA_downscale")
import ACTEA_common_tools_drift
import ACTEA_create_maps_and_animations
import ACTEA_gcs

__author__ = "Trond Kristiansen"
__email__ = "tk (at) actea.earth"
__created__ = datetime(2020, 6, 29)
__modified__ = datetime(2025, 9, 12)
__version__ = "1.0"
__status__ = (
    "Development, modified on 29.06.2020, 20.07.2020, 10.08.2020, "
    "11.12.2020, 26.03.2021, 28.03.2021, 21.12.2024, 05.03.2024, 10.03.2024,"
    "15.05.2024, 12.09.2025"
)


class Particle_Organizer:

    def __init__(self, project: str = "salmon"):
        self.project = project
        self.setup_fields()
        self.setup_logging()
        self.setup_GCS()

    def setup_fields(self):
        self.confobj: ActeaConf = ActeaConf(project=self.project)
        self.confobj.species = self.project
        self.individual_stations = False
        self.confobj.hatchery = None
        self.confobj.postfix = None
        self.start_time = datetime(2022, 3, 1)
        self.end_time = datetime(2022, 4, 30)

    def setup_logging(self):
        FORMAT = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
        logging.basicConfig(format=FORMAT)
        logging.root.handlers = []
        logging.root.setLevel("WARNING")
        self.logger = logging.getLogger("ACTEA-log")
        self.logger.setLevel(logging.INFO)

    def setup_GCS(self):
        ACTEA_bucket = "actea-shared"
        self.fs = gcsfs.GCSFileSystem(project="downscale")
        storage_client = storage.Client()
        self.gcs = ACTEA_gcs.ACTEA_gcs(frequency="monthly")
        self.bucket = storage_client.bucket(ACTEA_bucket)

    def setup_eggs(self):

        all_sites = {"ShelikofStrait":{
            "lat": 58.03086,
            "lon": -154.05555,
            "density": 1024,
            "start": self.start_time,
            "stop": self.end_time,
        }}

        self.confobj.st_lons.append(58.03086)
        self.confobj.st_lats.append(-154.05555)

        for i, site in enumerate(all_sites.keys()):
            self.logger.info(f"{site}: {all_sites[site]}")

        return all_sites

    def seed_elements(self, o, egg_site, i, site, radius):
        o.seed_elements(
            lon=egg_site["lon"],
            lat=egg_site["lat"],
            z="seafloor",
            density=egg_site["density"],
            radius=radius,
            radius_type="uniform",
            number=self.confobj.number_of_particles,
        #    terminal_velocity=self.confobj.terminal_velocity,
            origin_marker=i,
            time=[egg_site["start"], egg_site["stop"]],
            origin_marker_name=site,
        )
        self.logger.debug(f"[ACTEA_run_drift] Release schedule: {o.elements_scheduled}")
        self.logger.debug(
            f"[ACTEA_run_drift] Start release {egg_site['start']} and stop {egg_site['stop']} from {site}"
        )
        return o

    def run_simulation(self, o):
        o.run(
            duration=timedelta(hours=3 * 24 * 30),
            time_step=3600,
            time_step_output=timedelta(hours=1),
            export_variables=[
                "lat", "lon", "time", "trajectory", "z", "status", "age",
                "density", "sea_floor_depth_below_sea_level", "origin_marker",
                "stage_fraction", "hatched", "sea_water_temperature", 
                "sea_water_salinity"
            ],
            outfile=self.confobj.outputFilename,
        )

    def create_and_run_simulation(self, plot_only=False):
        self.confobj.number_of_particles = 2000
        all_sites = self.setup_eggs()

        for port_townsend_eggs in [True, False]:
            for radius in [0, 5000]:
                for density_type, dynamic_type in zip(["constant", "dynamic", "dynamic"], ["constant", "light", "dark"]):
            # for density_type, dynamic_type in zip(["constant"], ["constant"]):
            
                    if not plot_only:
                        for i, site in enumerate(all_sites.keys()):
                            egg_site = all_sites[site]

                            o = ACTEA_common.setup_and_config_oceandrift_module(self.fs, "walleye_pollock_eggs")
                            o = ACTEA_common.add_ocean_currents_glorys(o, self.fs, start_year=self.start_time.year, end_year=self.end_time.year)
                            o = ACTEA_common.add_wind_ERA5(o, self.fs, start_year=self.start_time.year, end_year=self.end_time.year)

                            o.port_townsend_eggs = port_townsend_eggs

                            if density_type == "constant":
                                o.constant_egg_density = True
                                o.dark_or_light_egg_density = dynamic_type
                            elif density_type == "dynamic":
                                o.constant_egg_density = False
                                o.dark_or_light_egg_density = dynamic_type
                            
                            o.__parallel_fail__ = True

                            if o.constant_egg_density:
                                postfix = f"constant_egg_density_seed_radius_{radius}"
                            else:
                                if o.dark_or_light_egg_density == "light":
                                    postfix = f"dynamic_egg_density_light_seed_radius_{radius}"
                                else:
                                    postfix = f"dynamic_egg_density_dark_seed_radius_{radius}"
                            
                            if o.port_townsend_eggs is True:
                                postfix = f"{postfix}_port_townsend_eggs"
                                print(f"Postfix is {postfix}")

                            self.confobj.postfix = postfix
                            self.confobj.create_output_filenames(plot_only=False,  start=self.start_time, end=self.end_time)

                            o = self.seed_elements(o, egg_site, i, site, radius)
                            self.run_simulation(o)

                            if not Path(f"{self.project}").exists():
                                Path(f"{self.project}").mkdir(parents=True, exist_ok=True)

                 #   ACTEA_create_maps_and_animations.make_map(self.confobj)

    def start_simulations(self):
        start_time = time.time()

        manual = False
        self.confobj.create_output_filenames(plot_only=manual, start=self.start_time, end=self.end_time)
        self.create_and_run_simulation(plot_only=manual)

        self.logger.debug(
            "---  It took %s seconds to run the script ---" % (time.time() - start_time)
        )


def main():
    run = Particle_Organizer(project="walleye_pollock_eggs")
    run.start_simulations()


if __name__ == "__main__":
    main()
