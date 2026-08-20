#!/usr/bin/env python
"""
| File: 11b_ardupilot_no_ros2.py
| Description: ROS 2-free variant of 11_ardupilot_multi_vehicle.py. Runs ArduPilot-
|              controlled multirotors in Isaac Sim without requiring rclpy / a ROS 2
|              install. Spawns 1 vehicle by default (raise NUM_VEHICLES for more).
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
    ArduPilotMavlinkBackend, ArduPilotMavlinkBackendConfig)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation

# Number of vehicles to spawn. Each one auto-launches its own ArduPilot SITL instance,
# so start with 1 for a first test and raise this once it works.
NUM_VEHICLES = 1

# Which built-in environment to load (see SIMULATION_ENVIRONMENTS in pegasus/simulator/params.py).
# Bright / open options that make the drone easy to see:
#   "Default Environment", "Flat Plane", "Warehouse", "Full Warehouse", "Simple Room"
# Darker ones: "Curved Gridroom", "Black Gridroom".
ENVIRONMENT = "Default Environment"

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Use the ArduPilot-tuned world settings. The standalone default is the px4 preset
        # (physics_dt = 1/250), which caps ArduPilot's main loop at 250 Hz and triggers
        # "PreArm: Main loop slow (250Hz < 400Hz)". The ardupilot preset runs physics at
        # 1/800 so the flight-controller loop comfortably clears its 400 Hz requirement.
        # This is what the GUI/extension mode does for the ArduPilot backend.
        self.pg.set_world_settings(physics_dt=1.0 / 800.0, rendering_dt=1.0 / 120.0)

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS[ENVIRONMENT])

        # Spawn the vehicles with the Ardupilot control backend, separated by 1.0 m along the x-axis
        for i in range(NUM_VEHICLES):
            self.vehicle_factory(i, gap_x_axis=1.0)

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def vehicle_factory(self, vehicle_id: int, gap_x_axis: float):
        """Auxiliar method to create multiple multirotor vehicles

        Args:
            vehicle_id (_type_): _description_
        """

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()

        # Create the multirotor configuration
        # ardupilot_autolaunch is False: we start ArduPilot SITL manually in a terminal
        # that has the `venv-ardupilot` virtualenv activated (that is where MAVProxy /
        # pymavlink / future live). Pegasus just opens the JSON FDM server (port 9002) and
        # the MAVLink udpin (port 14550) and waits for SITL to connect.
        backend_config = ArduPilotMavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "ardupilot_autolaunch": False,
            "ardupilot_dir": self.pg.ardupilot_path,
            "ardupilot_vehicle_model": "gazebo-iris",
            # ArduPilot's JSON FDM backend runs lockstep by default; match it here so the
            # sim and flight controller stay time-synced (this is what the GUI mode does).
            # Without it ArduPilot reports "main loop slow" and PreArm fails.
            "enable_lockstep": True
        })
        # ROS 2 backend intentionally omitted so no rclpy / ROS 2 install is required.
        config_multirotor.backends = [
            ArduPilotMavlinkBackend(config=backend_config),
        ]

        Multirotor(
            f"/World/drone{vehicle_id}",
            ROBOTS['Iris'],
            vehicle_id,
            [gap_x_axis * vehicle_id, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor
        )

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
