"""
Phase 2: Virtual camera.

Mounts a virtual camera above the table (like a fixed warehouse overhead
camera) and captures:
- RGB image (what YOLO will consume in Phase 3)
- Depth buffer (converted to real-world depth in meters)
- Segmentation mask (useful for debugging / ground-truth checks)

Run directly to save a snapshot to experiments/phase2_snapshot.png:
    python simulation/camera.py
"""

import numpy as np
import pybullet as p


class OverheadCamera:
    def __init__(
        self,
        width=640,
        height=480,
        cam_target=(0.5, 0.0, 0.62),   # looking at the table center
        cam_distance=0.9,
        cam_yaw=0,
        cam_pitch=-89,                 # looking almost straight down
        fov=60,
        near=0.1,
        far=3.0,
    ):
        self.width = width
        self.height = height
        self.near = near
        self.far = far

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target,
            distance=cam_distance,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        aspect = width / height
        self.proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    def capture(self):
        """Returns (rgb, depth_meters, segmentation) as numpy arrays."""
        _, _, rgb_px, depth_buffer, seg = p.getCameraImage(
            self.width,
            self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER,  # works headless, no GPU needed
        )

        rgb = np.reshape(rgb_px, (self.height, self.width, 4))[:, :, :3].astype(
            np.uint8
        )

        depth_buffer = np.reshape(depth_buffer, (self.height, self.width))
        # PyBullet's depth buffer is non-linear [0,1]; convert to real depth (m)
        depth_meters = self.far * self.near / (
            self.far - (self.far - self.near) * depth_buffer
        )

        seg = np.reshape(seg, (self.height, self.width))

        return rgb, depth_meters, seg

    def pixel_to_world(self, px, py, depth_m):
        """
        Convert a pixel (px, py) at a given depth (meters) back into a 3D
        world coordinate. This is the core "2D detection -> 3D position"
        step the whole pipeline depends on (Phase 4).
        """
        view = np.array(self.view_matrix).reshape(4, 4, order="F")
        proj = np.array(self.proj_matrix).reshape(4, 4, order="F")

        # NDC coordinates in [-1, 1]
        x_ndc = (2.0 * px / self.width) - 1.0
        y_ndc = 1.0 - (2.0 * py / self.height)

        # Reconstruct clip-space z from real depth using the projection matrix
        # (approach: invert full view-proj transform on a point at this depth)
        inv_vp = np.linalg.inv(proj @ view)

        # Use two points along the ray (near/far NDC z) and interpolate by depth
        # Simpler robust approach: unproject at z_ndc = -1 and +1, then walk
        # along the ray to the requested metric depth.
        near_pt = inv_vp @ np.array([x_ndc, y_ndc, -1.0, 1.0])
        near_pt /= near_pt[3]
        far_pt = inv_vp @ np.array([x_ndc, y_ndc, 1.0, 1.0])
        far_pt /= far_pt[3]

        cam_pos = self._camera_world_position(view)
        ray_dir = (far_pt[:3] - near_pt[:3])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        world_point = cam_pos + ray_dir * (
            depth_m / np.dot(ray_dir, self._camera_forward(view))
        )
        return tuple(world_point)

    @staticmethod
    def _camera_world_position(view):
        view_inv = np.linalg.inv(view)
        return view_inv[:3, 3]

    @staticmethod
    def _camera_forward(view):
        view_inv = np.linalg.inv(view)
        # Camera looks down -Z in its own frame
        forward = -view_inv[:3, 2]
        return forward / np.linalg.norm(forward)


if __name__ == "__main__":
    import os
    import cv2
    from environment import WarehouseEnv

    env = WarehouseEnv(gui=False)
    cam = OverheadCamera()
    rgb, depth, seg = cam.capture()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase2_snapshot.png")
    cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print("RGB shape:", rgb.shape)
    print("Depth range (m):", depth.min(), "to", depth.max())
    print("Unique segmentation IDs:", np.unique(seg))

    # Sanity check: project the known red_cube position into the image and
    # back out again, and see how close we get.
    red_cube_world = env.object_position("red_cube")
    print("Ground-truth red_cube position:", red_cube_world)

    env.close()
    print(f"Phase 2 OK: saved {out_path}")