"""
Phase 1: Core simulation environment.

Builds a PyBullet world containing:
- A ground plane
- A table
- A robot arm (KUKA iiwa, built into pybullet_data — no extra downloads needed)
- A few colored cube/box objects on the table

Run this file directly to sanity-check that everything loads:
    python simulation/environment.py
"""

import pybullet as p
import pybullet_data
import time


class WarehouseEnv:
    def __init__(self, gui: bool = False):
        """
        gui=True opens a visual PyBullet window (only works if you have a
        display). gui=False runs headless (DIRECT mode) — use this on
        servers / containers / CI, and for the vision pipeline later.
        """
        self.gui = gui
        connect_mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(connect_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.plane_id = None
        self.table_id = None
        self.robot_id = None
        self.object_ids = {}

        self._build_world()

    def _build_world(self):
        # Ground
        self.plane_id = p.loadURDF("plane.urdf")

        # Table — a simple custom thin tabletop rather than the bundled
        # pybullet_data asset. The bundled table's collision mesh geometry
        # isn't fully known/controllable, and testing showed the robot arm's
        # forearm link persistently intersects it regardless of target
        # height. A thin box we define ourselves lets us guarantee the arm
        # has real clearance to hover above it.
        self.table_top_z = 0.62
        table_half_extents = [0.35, 0.45, 0.02]  # thin slab, like a real tabletop
        table_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=table_half_extents
        )
        table_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.55, 0.4, 0.25, 1]
        )
        self.table_id = p.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.5, 0, self.table_top_z - table_half_extents[2]],
        )
        # Four thin legs, visual + collision, purely so it still looks like
        # a table -- positioned well clear of the arm's working column.
        leg_half = [0.02, 0.02, self.table_top_z / 2 - table_half_extents[2]]
        leg_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=leg_half)
        leg_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=leg_half, rgbaColor=[0.35, 0.25, 0.15, 1]
        )
        for dx in (-1, 1):
            for dy in (-1, 1):
                leg_x = 0.5 + dx * (table_half_extents[0] - 0.03)
                leg_y = dy * (table_half_extents[1] - 0.03)
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=leg_col,
                    baseVisualShapeIndex=leg_vis,
                    basePosition=[leg_x, leg_y, leg_half[2]],
                )

        # Robot arm — KUKA iiwa, also bundled, no download needed
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )

        # IMPORTANT: the arm's all-zero default pose points straight up and
        # physically intersects the table at this table position, sending
        # it flying past its own joint limits during the settle steps below.
        # Hover pose, well clear of the table (verified: >1cm clearance to
        # the tabletop at this height/joint configuration).
        self.HOME_POSE = [0.0, 0.26, 0.0, -0.64, 0.0, 2.16, 0.0]
        for j, target in enumerate(self.HOME_POSE):
            p.resetJointState(self.robot_id, j, target)
            p.setJointMotorControl2(
                self.robot_id, j, p.POSITION_CONTROL, targetPosition=target, force=300
            )

        # A couple of objects on the table top
        table_top_z = self.table_top_z
        self.object_ids["red_cube"] = self._spawn_cube(
            position=[0.45, 0.15, table_top_z + 0.03],
            half_extents=[0.025, 0.025, 0.025],
            rgba=[0.8, 0.1, 0.1, 1],
        )
        self.object_ids["blue_box"] = self._spawn_cube(
            position=[0.55, -0.15, table_top_z + 0.03],
            half_extents=[0.03, 0.02, 0.03],
            rgba=[0.1, 0.2, 0.8, 1],
        )

        # Let physics settle
        for _ in range(240):
            p.stepSimulation()

    def _spawn_cube(self, position, half_extents, rgba):
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba
        )
        body_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position,
        )
        return body_id

    def num_joints(self):
        return p.getNumJoints(self.robot_id)

    def object_position(self, name):
        pos, _ = p.getBasePositionAndOrientation(self.object_ids[name])
        return pos

    def step(self, n=1):
        for _ in range(n):
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def close(self):
        p.disconnect(self.client)


if __name__ == "__main__":
    env = WarehouseEnv(gui=True)
    print(f"Robot loaded with {env.num_joints()} joints")
    print(f"red_cube position:  {env.object_position('red_cube')}")
    print(f"blue_box position:  {env.object_position('blue_box')}")
    input("Press Enter to close...")
    env.close()
    print("Phase 1 OK: world builds and steps without errors.")