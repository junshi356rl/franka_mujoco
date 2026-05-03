import os
import numpy as np
from panda_mujoco_gym.envs.panda_env import FrankaEnv

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "push.xml")


class FrankaPushEnv(FrankaEnv):
    def __init__(
        self,
        reward_type,
        **kwargs,
    ):
        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=True,
            distance_threshold=0.05,
            goal_xy_range=0.15,
            obj_xy_range=0.1,
            goal_x_offset=0.2,
            goal_z_range=0.0,
            **kwargs,
        )

    def _set_action(self, action) -> None:
        action = action.copy()
        # for the pick and place task
        if not self.block_gripper:
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            fingers_ctrl = gripper_ctrl * 0.2
            fingers_width = self.get_fingers_width().copy() + fingers_ctrl
            fingers_half_width = np.clip(fingers_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1])

        elif self.block_gripper:
            pos_ctrl = action
            fingers_half_width = 0

        # control the gripper
        self.data.ctrl[-2:] = fingers_half_width

        # control the end-effector with mocap body
        pos_ctrl *= 0.05
        pos_ctrl += self.get_ee_position().copy()
        pos_ctrl[2] = 0.03

        self.set_mocap_pose(pos_ctrl, self.grasp_site_pose)