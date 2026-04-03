import numpy as np
import matplotlib.pyplot as plt

# 1: Generate 2000 random points between -5 and 5 
points = np.random.uniform(-5, 5, (2000, 2))


x = points[:, 0]
y = points[:, 1]


# 2:Calculating the mathematical function


partie1 = np.sin(np.sqrt(x**2 + y**2))
partie2 = 0.5 * np.cos(2*x + 2*y)
z = partie1 + partie2

# 3:Normalizing the data to the range [-1, 1]

x_min = np.min(x)
x_max = np.max(x)
x_norm = 2 * ((x - x_min) / (x_max - x_min)) - 1

y_min = np.min(y)
y_max = np.max(y)
y_norm = 2 * ((y - y_min) / (y_max - y_min)) - 1

z_min = np.min(z)
z_max = np.max(z)
z_norm = 2 * ((z - z_min) / (z_max - z_min)) - 1