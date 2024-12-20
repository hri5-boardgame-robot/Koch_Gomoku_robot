import mujoco
import numpy as np
import mink


class SimulatedRobot:
    def __init__(self, m, d, joint_name='joint6'):
        self.m = m
        self.d = d
        self.joint_name = joint_name

        frame_type = "body"
        self.end_effector_task = mink.FrameTask(
            frame_name=self.joint_name,
            frame_type=frame_type,
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0
        )
        self.tasks = [self.end_effector_task]
        self.configuration = mink.Configuration(self.m)
        self.configuration.update(self.d.qpos)
        self.limits = [mink.ConfigurationLimit(
            model=self.m)]  # enforce joint limits

    def _pos2pwm(self, pos: np.ndarray) -> np.ndarray:
        # pos in [-pi, pi], map to [0, 4096]
        return (pos / np.pi + 1.) * 2048

    def _pwm2pos(self, pwm: np.ndarray) -> np.ndarray:
        return (pwm / 2048.0 - 1.0) * np.pi

    def read_position(self) -> np.ndarray:
        # Return qpos in  pwm
        pos = self._pos2pwm(self.d.qpos[:6])

        return pos

    def forward_kinematics(self, qpos):
        self.d.qpos[:len(qpos)] = qpos
        mujoco.mj_fwdPosition(self.m, self.d)
        ee_body_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, self.joint_name)
        ee_pos = self.d.xpos[ee_body_id].copy()
        return ee_pos

    def inverse_kinematics_rot(self, ee_target_pos, ee_target_rot, rate=0.01, joint_name='joint6'):
        rotation = mink.SO3.from_matrix(ee_target_rot)
        T_wt = mink.SE3.from_rotation_and_translation(rotation, ee_target_pos)

        self.end_effector_task.set_target(T_wt)
        solver = "quadprog"
        pos_threshold = 1e-4
        ori_threshold = 1e-4
        dt = rate
        max_iters = 300

        for i in range(max_iters):
            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt,
                solver=solver,
                damping=1e-3,
                limits=[mink.ConfigurationLimit(self.m)]
            )

            self.configuration.integrate_inplace(vel, dt)
            err = self.end_effector_task.compute_error(self.configuration)
            if np.linalg.norm(err[:3]) <= pos_threshold and np.linalg.norm(err[3:]) <= ori_threshold:
                break

        q = self.configuration.q.copy()
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)
        return q
