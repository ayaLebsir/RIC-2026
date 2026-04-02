import numpy as np
import matplotlib.pyplot as plt

# --- 1: توليد النقط العشوائية ---
# هنا نولد 2000 نقطة عشوائية بين -5 و 5
points = np.random.uniform(-5, 5, (2000, 2))


x = points[:, 0]
y = points[:, 1]