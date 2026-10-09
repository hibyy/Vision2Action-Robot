import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
from environment import WarehouseEnv
from camera import OverheadCamera
from detector import detect_by_color

env = WarehouseEnv(gui=False)
cam = OverheadCamera()
rgb, depth, seg = cam.capture()

detections = detect_by_color(rgb, conf_threshold=0.3)
for det in detections:
    x, y, w, h = det["bbox"]
    patch = rgb[y:y+h, x:x+w]
    avg_color = patch.reshape(-1, 3).mean(axis=0)  # average R, G, B
    print(f"{det['label']:10s} conf={det['confidence']:.2f}  "
          f"avg_RGB=({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})")

env.close()