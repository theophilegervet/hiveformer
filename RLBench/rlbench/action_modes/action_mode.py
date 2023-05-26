from abc import abstractmethod

import numpy as np

from rlbench.action_modes.arm_action_modes import ArmActionMode
from rlbench.action_modes.gripper_action_modes import GripperActionMode
from rlbench.backend.scene import Scene
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK

class ActionMode(object):

    def __init__(self,
                 arm_action_mode: 'ArmActionMode',
                 gripper_action_mode: 'GripperActionMode'):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray, collision_checking=False):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass


class MoveArmThenGripper(ActionMode):
    """The arm action is first applied, followed by the gripper action. """

    def action(self, scene: Scene, action: np.ndarray, collision_checking=False):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        # DEBUG
        if type(self.arm_action_mode) is EndEffectorPoseViaIK:
            observations = self.arm_action_mode.action(scene, arm_action)
        else:
            observations = self.arm_action_mode.action(scene, arm_action, collision_checking=collision_checking)
        self.gripper_action_mode.action(scene, ee_action)
        return observations

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))
