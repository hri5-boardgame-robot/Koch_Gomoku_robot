
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from interface import SimulatedRobot
from robot import Robot, OperatingMode
from scipy.spatial.transform import Rotation as R
import json

# Default positions for illustrative purposes
HOME_POSITION = [-0.08, 0.07, 0.18]  # Example "home" hover position
DEFAULT_UP_Z = 0.14                  # "Safe" hover height
DEFAULT_DOWN_Z = 0.092               # Lower position for grasp/release


class OmokuBot:
    def __init__(self, use_real_robot=True, device_name='/dev/ttyACM0'):
        self.use_real_robot = use_real_robot
        self.device_name = device_name
        self.robot = None
        self.sim = None
        self.d = None
        self.m = None
        self.viewer = None

        # Workspace defaults
        self.x_max_distance = 0.075
        self.x_min_distance = -0.076
        self.y_max_distance = 0.24
        self.y_min_distance = 0.095
        self.reload_position = [-0.06915793,  0.06085899,  0.118]

        # Control parameters
        self.control_rate = 100  # Hz
        self.control_thread = None
        self.control_thread_running = False
        self.io_lock = threading.Lock()

        # Current target PWM
        self.current_target_pwm = None

        # Motion queue
        self.motion_queue = deque()

        self.robot_setup()
        print("ROBOT SETUP DONE")
        self.init_robot()

    def robot_setup(self):
        self.m = mujoco.MjModel.from_xml_path('low_cost_robot/scene.xml')
        self.d = mujoco.MjData(self.m)
        self.sim = SimulatedRobot(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)

        if self.use_real_robot:
            with self.io_lock:
                self.robot = Robot(device_name=self.device_name)
                # Enable torque and set position control mode
                self.robot._enable_torque()
                self.robot._set_position_control()
                self.robot.set_pi(3, 1000, 300)
                self.current_target_pwm = np.array(self.robot.read_position())
                print("Initial robot position (PWM):", self.current_target_pwm)
        else:
            self.current_target_pwm = np.array(self.sim.read_position())

    def init_robot(self):
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        self.start_control_thread()

        # Move to HOME_POSITION and wait until done
        self.move_ee_position_cartesian(HOME_POSITION, wait=True)
        # self.reload()
        if self.viewer is not None:
            self.viewer.sync()

    def start_control_thread(self):
        if self.control_thread_running:
            return
        self.control_thread_running = True
        self.control_thread = threading.Thread(
            target=self._control_loop, daemon=True)
        self.control_thread.start()

    def _control_loop(self):
        desired_period = 1.0 / self.control_rate
        while self.control_thread_running:
            start_loop = time.perf_counter()
            with self.io_lock:
                if self.use_real_robot:
                    current_robot_pwm = np.array(self.robot.read_position())
                else:
                    current_robot_pwm = np.array(self.current_target_pwm)

                current_qpos = self.sim._pwm2pos(current_robot_pwm)
                self.d.qpos[:6] = current_qpos[:6]
                mujoco.mj_forward(self.m, self.d)

                if self.motion_queue:
                    self.current_target_pwm = self.motion_queue.popleft()

                if self.current_target_pwm is not None:
                    pwm_cmd = np.array(self.current_target_pwm)
                    if self.use_real_robot:
                        self.robot.set_goal_pos(pwm_cmd)
                    else:
                        qpos_cmd = self.sim._pwm2pos(pwm_cmd)
                        self.d.qpos[:6] = qpos_cmd[:6]
                        mujoco.mj_forward(self.m, self.d)

            end_loop = time.perf_counter()
            elapsed = end_loop - start_loop
            sleep_time = desired_period - elapsed
            if self.viewer is not None:
                # Update viewer to reflect new positions
                self.viewer.sync()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop_control_thread(self):
        self.control_thread_running = False
        if self.control_thread:
            self.control_thread.join()

    def wait_for_motion_done(self, timeout=10.0):
        """
        Wait until the motion queue is empty or until timeout seconds pass.
        This ensures the robot has finished executing the currently queued motions.
        """
        start_time = time.time()
        while True:
            with self.io_lock:
                if not self.motion_queue:
                    return
            if time.time() - start_time > timeout:
                print("Timeout waiting for motion to complete!")
                return
            time.sleep(0.01)

    def move_ee_position_cartesian(self, destination, move_time=2.0, wait=True):
        destination = np.array(destination)
        target_ee_rot = R.from_euler('x', -90, degrees=True).as_matrix()

        current_xyz = self.get_ee_xyz()
        start_xyz = np.array(current_xyz)

        steps = int(self.control_rate * move_time)
        if steps < 1:
            steps = 1

        waypoints = []
        for i in range(steps + 1):
            alpha = i / steps
            interp_xyz = start_xyz * (1 - alpha) + destination * alpha
            waypoints.append(interp_xyz)

        with self.io_lock:
            if self.use_real_robot:
                current_positions = np.array(self.robot.read_position())
            else:
                current_positions = np.array(self.current_target_pwm)

            gripper_pwm = current_positions[5] if len(
                current_positions) > 5 else 2500

            self.motion_queue.clear()
            for xyz in waypoints:
                qpos_ik = self.sim.inverse_kinematics_rot(
                    xyz, target_ee_rot, rate=0.001, joint_name='joint6')
                if qpos_ik is None:
                    print("IK failed for waypoint:", xyz)
                    continue
                pwm_values = self.sim._pos2pwm(qpos_ik[:5]).astype(int)
                full_pwm_values = np.concatenate((pwm_values, [gripper_pwm]))
                self.motion_queue.append(full_pwm_values)
            print(
                f"Enqueued {len(self.motion_queue)} steps for Cartesian move.")

        if wait:
            self.wait_for_motion_done()

    def gripper(self, mode, move_time=1.0, wait=True):
        with self.io_lock:
            if self.current_target_pwm is None:
                if self.use_real_robot:
                    self.current_target_pwm = np.array(
                        self.robot.read_position())
                else:
                    self.current_target_pwm = np.array(
                        self.sim.read_position())

            current_positions = np.array(self.current_target_pwm)
            if mode == "open":
                gripper_target = 2200
            elif mode == "close":
                gripper_target = 1965
            elif mode == "half_open":
                gripper_target = 2070
            else:
                gripper_target = current_positions[5]

            steps = int(self.control_rate * move_time)
            if steps < 1:
                steps = 1

            start_val = current_positions[5]
            end_val = gripper_target
            self.motion_queue.clear()

            for i in range(steps + 1):
                alpha = i / steps
                g_val = int(start_val * (1 - alpha) + end_val * alpha)
                pwm_step = current_positions.copy()
                pwm_step[5] = g_val
                self.motion_queue.append(pwm_step)
            print(f"Enqueued {len(self.motion_queue)} steps for gripper move.")

        if wait:
            self.wait_for_motion_done()

    def get_ee_xyz(self):
        with self.io_lock:
            if self.use_real_robot:
                positions = np.array(self.robot.read_position())
            else:
                positions = np.array(self.current_target_pwm)
            current_qpos = self.sim._pwm2pos(positions)
            current_ee_xyz = self.sim.forward_kinematics(current_qpos)
        return current_ee_xyz

    def grid_to_xyz(self, grid_y, grid_x, z_plane=0.14):
        if not (0 <= grid_x < 9 and 0 <= grid_y < 9):
            raise ValueError("Grid coordinates must be between 0 and 8")
        x_position = self.x_min_distance + \
            (self.x_max_distance - self.x_min_distance) * (grid_x / 8)
        y_position = self.y_min_distance + \
            (self.y_max_distance - self.y_min_distance) * (grid_y / 8)
        return np.array([x_position, y_position, z_plane])

    def move_to_grid(self, grid_y, grid_x, z_plane=0.14, wait=True):
        destination = self.grid_to_xyz(grid_y, grid_x, z_plane)
        self.move_ee_position_cartesian(destination, wait=wait)

    def grasp(self, wait=True):
        self.gripper("close", wait=wait)

    def release(self, wait=True):
        self.gripper("open", wait=wait)

    def release_half(self, wait=True):
        self.gripper("half_open", wait=wait)

    def move_up(self, z=DEFAULT_UP_Z, wait=True):
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        self.move_ee_position_cartesian([x, y, z], wait=wait)

    def move_down(self, z=DEFAULT_DOWN_Z, wait=True):
        current_xyz = self.get_ee_xyz()
        x, y, _ = current_xyz
        self.move_ee_position_cartesian([x, y, z], wait=wait)

    def reload(self):
        self.move_ee_position_cartesian(HOME_POSITION)
        x, y, z = self.reload_position
        self.move_ee_position_cartesian([x, y, DEFAULT_UP_Z])
        self.release()
        self.move_ee_position_cartesian([x, y, z])
        self.grasp()
        self.move_ee_position_cartesian(HOME_POSITION)

        print("reload done")

    def __del__(self):
        self.stop_control_thread()
