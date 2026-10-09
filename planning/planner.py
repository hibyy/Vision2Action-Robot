"""
Phase 7: Motion planning (minimal RRT, joint space).

Instead of us hand-picking waypoints (hover -> approach -> descend, as in
Phase 5/6), this searches for a collision-free path through joint space
on its own:

    1. Grow a tree of joint configurations, starting from the arm's
       current pose.
    2. At each step, sample a random configuration (occasionally the
       actual goal, to bias the search toward finishing), find the
       nearest tree node, and step a bounded distance toward it.
    3. Only keep the new node if that configuration is collision-free.
    4. Stop once a node lands within `goal_tolerance` of the goal.

This is intentionally the minimal version: no path smoothing, no
optimization -- the point is to prove the planner finds *a* valid path
around obstacles on its own, not to make that path pretty. Path
smoothing is a natural, well-scoped Phase 7b if you want it later.

Collision checking reuses the exact technique from the Phase 5 debugging
session: temporarily set the arm to a candidate configuration and check
`getClosestPoints` against obstacles, then restore whatever configuration
the arm was actually in before the check (the planner must never leave
side effects on the live simulation while it's just *searching*).
"""

import math
import random

import pybullet as p


class RRTPlanner:
    def __init__(self, robot_id, joint_limits, obstacle_ids, ee_link=6, clearance=0.005):
        """
        robot_id:      the arm's PyBullet body id
        joint_limits:  list of (lower, upper) tuples, one per joint
        obstacle_ids:  list of body ids to avoid (e.g. [table_id])
        clearance:     minimum allowed distance (m) to any obstacle
        """
        self.robot_id = robot_id
        self.joint_limits = joint_limits
        self.num_joints = len(joint_limits)
        self.obstacle_ids = obstacle_ids
        self.ee_link = ee_link
        self.clearance = clearance

    # ---- collision checking -------------------------------------------------

    def _get_joint_state(self):
        return [p.getJointState(self.robot_id, j)[0] for j in range(self.num_joints)]

    def _set_joint_state(self, config):
        for j, val in enumerate(config):
            p.resetJointState(self.robot_id, j, val)

    def is_collision_free(self, config):
        """
        Temporarily poses the arm at `config`, checks for collisions against
        every obstacle, then restores whatever pose the arm was actually in.
        The simulation is left exactly as it was found.
        """
        saved_state = self._get_joint_state()

        self._set_joint_state(config)
        p.performCollisionDetection()

        collision = False
        for obstacle_id in self.obstacle_ids:
            contacts = p.getClosestPoints(
                self.robot_id, obstacle_id, distance=self.clearance
            )
            if contacts:
                collision = True
                break

        self._set_joint_state(saved_state)  # restore -- planner must be a pure search
        return not collision

    def is_path_segment_free(self, from_config, to_config, resolution=0.05):
        """
        Checks not just the endpoint but points along the straight-line path
        in joint space between from_config and to_config. Without this, a
        step could land on two individually collision-free configurations
        while still sweeping through an obstacle in between -- a classic
        RRT gap, especially with a thin obstacle like our table.
        """
        dist = self._distance(from_config, to_config)
        num_checks = max(1, int(math.ceil(dist / resolution)))
        for i in range(1, num_checks + 1):
            t = i / num_checks
            interp = [
                f + (t_ - f) * t for f, t_ in zip(from_config, to_config)
            ]
            if not self.is_collision_free(interp):
                return False
        return True

    # ---- RRT core -------------------------------------------------------------

    def _random_config(self):
        return [random.uniform(lo, hi) for lo, hi in self.joint_limits]

    def _distance(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _steer(self, from_config, to_config, max_step):
        dist = self._distance(from_config, to_config)
        if dist <= max_step:
            return list(to_config)
        ratio = max_step / dist
        return [
            f + (t - f) * ratio for f, t in zip(from_config, to_config)
        ]

    def plan(self, start_config, goal_config, max_iters=3000, max_step=0.25,
              goal_tolerance=0.08, goal_bias=0.1):
        """
        Returns a list of joint configurations from start to goal (inclusive),
        or None if no path was found within max_iters.
        """
        if not self.is_collision_free(start_config):
            raise ValueError("start_config is already in collision")
        if not self.is_collision_free(goal_config):
            raise ValueError("goal_config is in collision -- pick a different target")

        tree_nodes = [list(start_config)]
        parent = {0: None}

        for _ in range(max_iters):
            sample = goal_config if random.random() < goal_bias else self._random_config()

            nearest_idx = min(
                range(len(tree_nodes)),
                key=lambda i: self._distance(tree_nodes[i], sample),
            )
            nearest = tree_nodes[nearest_idx]

            new_config = self._steer(nearest, sample, max_step)

            if not self.is_path_segment_free(nearest, new_config):
                continue

            tree_nodes.append(new_config)
            new_idx = len(tree_nodes) - 1
            parent[new_idx] = nearest_idx

            if self._distance(new_config, goal_config) < goal_tolerance:
                if not self.is_path_segment_free(new_config, goal_config):
                    continue
                tree_nodes.append(list(goal_config))
                parent[len(tree_nodes) - 1] = new_idx
                return self._extract_path(tree_nodes, parent, len(tree_nodes) - 1)

        return None  # no path found within the iteration budget

    def _extract_path(self, tree_nodes, parent, goal_idx):
        path = []
        idx = goal_idx
        while idx is not None:
            path.append(tree_nodes[idx])
            idx = parent[idx]
        path.reverse()
        return path


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
    from environment import WarehouseEnv   # noqa: E402
    from robot import RobotController      # noqa: E402

    env = WarehouseEnv(gui=True)
    robot = RobotController(env)

    joint_limits = list(zip(robot.lower_limits, robot.upper_limits))
    planner = RRTPlanner(
        robot_id=env.robot_id,
        joint_limits=joint_limits,
        obstacle_ids=[env.table_id],
    )

    start_config = [p.getJointState(env.robot_id, j)[0] for j in range(7)]

    cube_pos = env.object_position("red_cube")
    approach_target = (cube_pos[0], cube_pos[1], cube_pos[2] + 0.15)
    orn = p.getQuaternionFromEuler([0, 3.14159, 0])
    goal_config = list(p.calculateInverseKinematics(
        env.robot_id, 6, approach_target, targetOrientation=orn,
        lowerLimits=robot.lower_limits, upperLimits=robot.upper_limits,
        jointRanges=robot.joint_ranges, restPoses=start_config,
        maxNumIterations=200, residualThreshold=1e-4,
    ))

    print("Planning path from current pose to above red_cube...")
    print("Start config:", [round(x, 3) for x in start_config])
    print("Goal config: ", [round(x, 3) for x in goal_config])

    path = planner.plan(start_config, goal_config)

    if path is None:
        print("FAILED: no collision-free path found.")
    else:
        print(f"SUCCESS: found a path with {len(path)} waypoints.")

        # Sanity-check: every waypoint AND every segment between consecutive
        # waypoints is collision-free (not just the endpoints)
        all_clear = all(planner.is_collision_free(cfg) for cfg in path)
        segments_clear = all(
            planner.is_path_segment_free(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )
        print(f"All waypoints collision-free: {all_clear}")
        print(f"All segments (interpolated) collision-free: {segments_clear}")

        # Execute it: drive through each waypoint with position control
        for cfg in path:
            for j, val in enumerate(cfg):
                p.setJointMotorControl2(
                    env.robot_id, j, p.POSITION_CONTROL, targetPosition=val, force=300
                )
            env.step(30)

        final_ee = robot.end_effector_position()
        print(f"Final end-effector position after executing path: "
              f"({final_ee[0]:+.3f}, {final_ee[1]:+.3f}, {final_ee[2]:+.3f})")
        print(f"Target was:                                        "
              f"({approach_target[0]:+.3f}, {approach_target[1]:+.3f}, {approach_target[2]:+.3f})")

    env.close()