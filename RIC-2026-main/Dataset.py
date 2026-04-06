import numpy as np
import matplotlib.pyplot as plt



# 1:Calculating the mathematical function-----------------

def f(x, y):
    # f(x, y) = sin(sqrt(x^2 + y^2)) + 0.5*cos(2x + 2y)
    r = np.sqrt(x**2 + y**2)
    return np.sin(r) + 0.5 * np.cos(2*x + 2*y)


# 2: Generate 2000 random points between -5 and 5 --------

# Paramètres
n_points = 2000
x_min, x_max = -5, 5
y_min, y_max = -5, 5

# Générer des points aléatoires
np.random.seed(42)
x = np.random.uniform(x_min, x_max, n_points)
y = np.random.uniform(y_min, y_max, n_points)

# Calculer z
z = f(x, y)






# 3:Normalizing the data to the range [-1, 1]-----------

# X (entrées)
X = np.column_stack([x, y])
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# y (sorties)
z_mean = np.mean(z)
z_std = np.std(z)
z_norm = (z - z_mean) / z_std





# 4:l'affichage---------------------------------------

plt.figure(figsize=(10, 8))

# Nuage de points 3D
ax = plt.axes(projection='3d')
scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=10, alpha=0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Dataset - Vérité Terrain (2000 points)')


plt.show()
