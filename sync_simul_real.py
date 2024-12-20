import time
import numpy as np
import mujoco
import mujoco.viewer

from interface import SimulatedRobot
from robot import Robot, OperatingMode

# Set this to 'FK' for forward kinematics testing or 'IK' for inverse kinematics testing
TEST_MODE = 'FK'

# Load the MuJoCo model
# m = mujoco.MjModel.from_xml_path(
#     './low_cost_robot/low_cost_robot.xml')
m = mujoco.MjModel.from_xml_path(
    'low_cost_robot/scene.xml')
d = mujoco.MjData(m)


# Initialize simulated robot
r = SimulatedRobot(m, d)
# Current positions: [3084 2038 1984 1013 2036 1969]
# Initialize real robot
robot = Robot(device_name='/dev/ttyACM0')
# robot = Robot(device_name='/dev/tty.usbmodem58760435301')

# Read initial positions and set initial simulation state
positions = robot.read_position()
positions = np.array(positions)
d.qpos[:6] = r._pwm2pos(positions)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        if TEST_MODE == 'FK':
            # Read the current positions from the real robot
            positions = robot.read_position()
            positions = np.array(positions)
            print(f"Current positions: {positions}")
            current_qpos = r._pwm2pos(positions)  # Adjust if necessary

            # Compute forward kinematics to get the current end-effector position
            current_ee_pos = r.forward_kinematics(current_qpos)

            # Display the current end-effector position
            print(f"Current end-effector position: {current_ee_pos}")

        elif TEST_MODE == 'IK':
            # Define the target end-effector position
            target_ee_pos = np.array([0.04745835,  0.15619665,  0.05095877])
            current_position = robot.read_position()
            current_position = np.array(current_position)
            current_position = r._pwm2pos(current_position)
            d.qpos[:6] = current_position

            # Iteratively compute the inverse kinematics to get joint positions
            max_iterations = 100
            tolerance = 1e-2
            for _ in range(max_iterations):
                qpos_ik = r.inverse_kinematics(current_position,
                                               target_ee_pos, rate=0.1, joint_name='joint6')
                d.qpos[:6] = qpos_ik
                current_position = qpos_ik
                mujoco.mj_forward(m, d)
                pwm_values = r._pos2pwm(qpos_ik).astype(int)

                time.sleep(0.02)

                # Check for convergence
                ee_pos = d.xpos[mujoco.mj_name2id(
                    m, mujoco.mjtObj.mjOBJ_BODY, 'joint6')]
                error = np.linalg.norm(target_ee_pos - ee_pos)
                if error < tolerance:
                    print(f"Converged to target position with error: {error}")
                    break

            # Convert joint positions to PWM values and ensure they are integers
            pwm_values = r._pos2pwm(qpos_ik).astype(int)

            # Visualize each interpolated position in the simulation
            current_position = robot.read_position()
            interpolated_positions = robot.get_interpolate_pose(
                current_position, pwm_values)
            for positions in interpolated_positions:
                # Convert PWM positions back to joint angles
                positions = np.array(positions)
                robot.set_goal_pos(positions)
                print(f"Setting goal position: {positions}")
                current_qpos = r._pwm2pos(positions)

                # Update simulation state
                d.qpos[:6] = current_qpos

                # Step and render
                mujoco.mj_forward(m, d)
                viewer.sync()

                # Add small delay for visualization
                time.sleep(0.02)

            # Disable torque after reaching the target position
            robot._disable_torque()

        # Step the simulation
        mujoco.mj_step(m, d)
        viewer.sync()

        # Sleep to maintain real-time simulation
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
