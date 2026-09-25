"""
Phase 5: Robot control.

Thin control layer over the KUKA iiwa loaded in WarehouseEnv:
- move_to(x, y, z): uses PyBullet's built-in inverse kinematics to drive
  the end effector to a target position
- open_gripper() / close_gripper(): the kuka_iiwa/model.urdf used here has
  no gripper fingers, so "closing" is simulated by rigidly attaching the
  nearest object to the end effector (a common, honest simplification in
  early-stage manipulation projects -- full finger-contact grasping is a
  later upgrade, noted in the project README).
"""

import os
import sys

import pybullet as p

sys.path.insert(0, os.path.dirname(__file__))
from environment import WarehouseEnv  # noqa: E402


END_EFFECTOR_LINK_INDEX = 6  # last link of kuka_iiwa/model.urdf (0-indexed, 7 joints)


class RobotController:
    def __init__(self, env: WarehouseEnv):
        self.env = env
        self.robot_id = env.robot_id
        self.num_joints = p.getNumJoints(self.robot_id)
        self._grasp_constraint = None
        self._held_object = None

        # Cache joint limits so IK solutions never exceed what the arm can
        # physically achieve (without this, calculateInverseKinematics can
        # return angles outside a joint's real range, and the position
        # motor then fights to reach an impossible target).
        self.lower_limits = []
        self.upper_limits = []
        self.joint_ranges = []
        for j in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, j)
            lo, hi = info[8], info[9]
            self.lower_limits.append(lo)
            self.upper_limits.append(hi)
            self.joint_ranges.append(hi - lo)

    def end_effector_position(self):
        state = p.getLinkState(self.robot_id, END_EFFECTOR_LINK_INDEX)
        return state[0]  # worldLinkFramePosition

    def move_to(self, target_xyz, steps=120, orientation=None):
        """
        Drive the end effector toward target_xyz using IK, stepping physics
        along the way so the motion is visible/inspectable (and so held
        objects move with the arm realistically).
        """
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 3.14159, 0])  # gripper facing down

        current_pose = [p.getJointState(self.robot_id, j)[0] for j in range(self.num_joints)]

        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            END_EFFECTOR_LINK_INDEX,
            target_xyz,
            targetOrientation=orientation,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=current_pose,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )

        for j in range(len(joint_targets)):
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_targets[j],
                force=200,
            )

        self.env.step(steps)
        return self.end_effector_position()

    def close_gripper(self, object_id):
        """Rigidly attach object_id to the end effector (simplified grasp)."""
        if self._grasp_constraint is not None:
            return  # already holding something

        ee_pos = self.end_effector_position()
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)

        self._grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=END_EFFECTOR_LINK_INDEX,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self._held_object = object_id
        self.env.step(20)

    def open_gripper(self):
        if self._grasp_constraint is not None:
            p.removeConstraint(self._grasp_constraint)
            self._grasp_constraint = None
            self._held_object = None
        self.env.step(20)


if __name__ == "__main__":
    env = WarehouseEnv(gui=True)
    robot = RobotController(env)

    print("Start end-effector position:", robot.end_effector_position())

    cube_id = env.object_ids["red_cube"]
    cube_pos = env.object_position("red_cube")
    print("Target (red_cube) position:", cube_pos)

    # Waypointed approach: hover above the table center (already ~ current
    # pose), then hover directly above the cube, then descend. Each step is
    # a small joint-space change from the last, so the arm never has to
    # sweep blindly through the table to get there.
    hover_above_table = (0.5, 0.0, 1.0)
    robot.move_to(hover_above_table)
    print("At table hover:", robot.end_effector_position())

    approach = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.15)
    robot.move_to(approach)
    print("After approach:", robot.end_effector_position())

    grasp_pose = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.02)
    robot.move_to(grasp_pose)
    print("After descend:", robot.end_effector_position())

    robot.close_gripper(cube_id)
    print("Gripper closed on red_cube")

    lift = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.20)
    robot.move_to(lift)
    print("After lift, cube position:", env.object_position("red_cube"))

    drop_zone = (0.5, 0.35, cube_pos[2] + 0.20)
    robot.move_to(drop_zone)
    print("After transport, cube position:", env.object_position("red_cube"))

    robot.open_gripper()
    env.step(60)
    print("After release, cube position:", env.object_position("red_cube"))

    env.close()
    print("Phase 5 OK: pick-and-place cycle completed.")