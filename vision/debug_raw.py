import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
from environment import WarehouseEnv
from camera import OverheadCamera
from detector import _get_model
from detector import _get_model
model = _get_model()
print(model.names)
env = WarehouseEnv(gui=True)
cam = OverheadCamera()
rgb, depth, seg = cam.capture()

model = _get_model()
results = model.predict(rgb, conf=0.001, verbose=False)[0]  # near-zero threshold

print(f"{len(results.boxes)} raw boxes (no filtering):")
for box in results.boxes:
    cls_id = int(box.cls[0])
    print(f"  {model.names[cls_id]:12s} conf={float(box.conf[0]):.3f}")

env.close()