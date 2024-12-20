import time
import numpy as np
import mujoco
import mujoco.viewer


class OmokuBot:
    def __init__(self, use_real_robot=False):
        self.use_real_robot = use_real_robot
        self.robot = None
        self.r = None
        self.d = None
        self.m = None
        self.viewer = None
        self.positions = None

        # Robot setup
        self.robot_setup()

    def robot_setup(self):
        self.m = mujoco.MjModel.from_xml_path(
            'omoku_bot/low_cost_robot/scene.xml')
        self.d = mujoco.MjData(self.m)

    def init_robot(self):
        # Use the standard launch function to get interactive controls
        self.viewer = mujoco.viewer.launch_passive(
            model=self.m,
            data=self.d
        )

    def run_simulation(self):
        if self.viewer is not None:
            while self.viewer.is_running():
                self.viewer.sync()
        else:
            print("Viewer not initialized. Call init_robot() first.")


if __name__ == "__main__":
    robot = OmokuBot(use_real_robot=False)
    robot.init_robot()
    robot.run_simulation()
