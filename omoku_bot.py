import time
# import keyboard

import numpy as np
import mujoco
import mujoco.viewer
from interface import SimulatedRobot
from robot import Robot, OperatingMode
from scipy.spatial.transform import Rotation as R

# Default positions for illustrative purposes
HOME_POSITION = [0.0, 0.16, 0.14]          # Example "home" hover position
TAKE_PICK_POSITION = [-0.08, 0.085, 0.14]  # Example pick position
DEFAULT_UP_Z = 0.14                        # "Safe" hover height
DEFAULT_DOWN_Z = 0.07                      # Lower position for grasp/release


class OmokuBot:
    def __init__(self, use_real_robot=True):
        """
        Initialize the OmokuBot class, set up the robot, and define workspace boundaries.
        If use_real_robot=False, code will run simulation only.
        """
        self.use_real_robot = use_real_robot
        self.robot = None
        self.sim = None
        self.d = None
        self.m = None
        self.viewer = None
        self.position_pwm = None

        # Robot setup
        self.robot_setup()

        if self.use_real_robot:
            self.position_pwm = self.robot.read_position()
            
        else:
            # If no real robot, mock initial positions or use default
            self.position_pwm = np.array([3084, 2038, 1984, 1013, 2036, 2500])

        # Additional parameters
        self.telescope_rate = 1.5e-2
        self.tolerance = 1e-5
        self.x_max_distance = 0.07
        self.x_min_distance = -0.07
        self.y_max_distance = 0.24
        self.y_min_distance = 0.10

        print("ROBOT SETUP DONE")

    def robot_setup(self):
        """
        Set up the Mujoco model and SimulatedRobot interface, and connect to the real robot if enabled.
        """
        self.m = mujoco.MjModel.from_xml_path(
            'low_cost_robot/scene.xml')
        self.d = mujoco.MjData(self.m)
        self.sim = SimulatedRobot(self.m, self.d)

        if self.use_real_robot:
            self.robot = Robot(device_name='/dev/ttyACM0')
        else:
            # No real robot: robot remains None
            self.robot = None
            self.position_pwm = self.sim.read_position()

    def init_robot(self):
        """
        Initialize the robot viewer and move to a default position.
        """
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        # Ensure gripper is open at start
        self.release()
        self.move_ee_position(TAKE_PICK_POSITION)
        time.sleep(10)
        if self.viewer is not None:
            self.viewer.sync()

    def move_joint(self, positions):
        """
        Move the robot joints to the given PWM positions with interpolation.
        """
        if self.use_real_robot:
            pwm_values = np.array(positions)
            current_position = self.robot.read_position()
            interpolated_positions = self.robot.get_interpolate_pose(
                current_position, pwm_values
            )

            for pos in interpolated_positions:
                pos = np.array(pos)
                self.robot.set_goal_pos(pos)
                current_qpos = self.sim._pwm2pos(pos)
                self.d.qpos[:6] = current_qpos[:6]
                mujoco.mj_forward(self.m, self.d)
                mujoco.mj_step(self.m, self.d)
                time.sleep(0.01)
        else:
            # Simulation-only: set qpos directly
            current_qpos = self.sim.read_position()
            conveted_pose = self.sim._pwm2pos(positions)
            self.d.qpos[:6] = conveted_pose[:6]
            mujoco.mj_forward(self.m, self.d)

    def move_ee_position(self, destination):
        destination = np.array(destination)
        target_ee_rot = R.from_euler('x', -90, degrees=True).as_matrix()
        # target_ee_rot = target_ee_rot @ \
        #     R.from_euler('y', 45, degrees=True).as_matrix()

        current_ee_pos = self.d.xpos[mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, 'joint6')]

        # Interpolate between current_ee_pos and destination
        num_waypoints = 100
        waypoints = np.linspace(current_ee_pos, destination, num_waypoints)

        for waypoint in waypoints:
            qpos_ik = self.sim.inverse_kinematics_rot(
                waypoint, target_ee_rot, rate=0.05, joint_name='joint6'
            )

            # Now qpos_ik is the IK solution after iteration inside inverse_kinematics_rot
            self.d.qpos[:5] = qpos_ik[:5]
            mujoco.mj_forward(self.m, self.d)
            time.sleep(0.02)
            if self.use_real_robot:
                # Convert joint positions to PWM values
                pwm_values = self.sim._pos2pwm(qpos_ik[:5]).astype(int)
                current_gripper_pwm = self.position_pwm[5]
                full_pwm_values = np.concatenate(
                    (pwm_values, [current_gripper_pwm]))
                # print(full_pwm_values, type(full_pwm_values[0]))
                full_pwm_values = [int(i) for i in full_pwm_values]
                print(full_pwm_values, type(full_pwm_values[0]))

                # raise KeyError
                self.robot.set_goal_pos(full_pwm_values)
                time.sleep(0.02)

                # Update simulation from actual robot position
                current_position = self.robot.read_position()
                converted_positions = self.sim._pwm2pos(
                    np.array(current_position))
                self.d.qpos[:6] = converted_positions[:6]
            else:
                # Simulation-only
                self.d.qpos[:5] = qpos_ik[:5]

            mujoco.mj_forward(self.m, self.d)

            if self.viewer is not None:
                self.viewer.sync()

    def gripper(self, mode):
        """
        Control the gripper.
        mode: "open" or "close"
        """
        if mode == "open":
            self.position_pwm[5] = 2200
        elif mode == "close":
            self.position_pwm[5] = 1965

        # For simulation-only mode, just update simulation state
        # If use_real_robot, also send commands to hardware
        self._move_gripper()

    def _move_gripper(self):
        """
        Internal method to move the gripper to the currently set positions[5].
        """
        if self.use_real_robot:
            pwm_values = np.array(self.position_pwm)
            current_position = self.robot.read_position()
            interpolated_positions = self.robot.get_interpolate_pose(
                current_position, pwm_values
            )

            for pos in interpolated_positions:
                pos = np.array(pos)
                self.robot.set_goal_pos(pos)
                current_qpos = self.sim._pwm2pos(pos)
                self.d.qpos[:6] = current_qpos[:6]
                mujoco.mj_forward(self.m, self.d)
                mujoco.mj_step(self.m, self.d)
                time.sleep(0.01)
        else:
            # Simulation-only: update qpos directly

            target_qpos = self.position_pwm
            current_qpos = self.sim.read_position()

            print("target_qpos", target_qpos)
            print("current_qpos", current_qpos)

            self.d.qpos[:6] = self.sim._pwm2pos(target_qpos)[:6]
            mujoco.mj_forward(self.m, self.d)
            if self.viewer is not None:
                self.viewer.sync()
            time.sleep(0.02)

    def get_ee_xyz(self):
        """
        Get the current end-effector XYZ position.
        If running simulation-only, returns simulated EE position.
        """
        if self.use_real_robot:
            positions = np.array(self.robot.read_position())
        else:
            # Simulation-only: Use stored positions or current qpos
            positions = self.sim.read_position()

        current_qpos = self.sim._pwm2pos(positions)
        current_ee_xyz = self.sim.forward_kinematics(current_qpos)
        return current_ee_xyz

    def get_joint_rad(self):
        """
        Get the current joint angles (in radians).
        """
        if self.use_real_robot:
            positions = np.array(self.robot.read_position())
        else:
            positions = self.position_pwm
        current_qpos = self.sim._pwm2pos(positions)
        return current_qpos

    def set_workspace(self, x_min, x_max, y_min, y_max):
        """
        Set workspace boundaries.
        """
        self.x_min_distance = x_min
        self.x_max_distance = x_max
        self.y_min_distance = y_min
        self.y_max_distance = y_max

    def move_to_grid(self, grid_y, grid_x, z_plane=0.14):
        """
        Move the robot to a specific grid position on a 9x9 grid.
        grid_x, grid_y: integer coordinates between 0 and 8
        z_plane: target z height
        """
        if not (0 <= grid_x < 9 and 0 <= grid_y < 9):
            raise ValueError("Grid coordinates must be between 0 and 8")

        # Define workspace bounds
        x_min, x_max = self.x_min_distance, self.x_max_distance
        y_min, y_max = self.y_min_distance, self.y_max_distance

        # Calculate the grid position
        x_position = x_min + (x_max - x_min) * (grid_x / 8)
        y_position = y_min + (y_max - y_min) * (1-(grid_y / 8))
        destination = np.array([x_position, y_position, z_plane])

        self.move_ee_position(destination)

    def grasp(self):
        # Example: just close gripper
        self.position_pwm = self.sim.read_position()
        self.gripper("close")

    def release(self):
        # Example: just open gripper
        self.position_pwm = self.sim.read_position()
        print(self.position_pwm)
        self.gripper("open")
        time.sleep(1000)


    def move_up(self):
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        self.move_ee_position([x, y, DEFAULT_UP_Z])

    def move_down(self):
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        self.move_ee_position([x, y, DEFAULT_DOWN_Z])

# Example usage (uncomment if you want to run this standalone):
# if __name__ == "__main__":
#     # Set use_real_robot=False to run simulation only
#     robot = OmokuBot(use_real_robot=False)
#     robot.init_robot()
#     robot.set_workspace(-0.07, 0.07, 0.10, 0.24)
#     robot.move_to_grid(4, 4)
#     robot.grasp()
#     robot.release()
