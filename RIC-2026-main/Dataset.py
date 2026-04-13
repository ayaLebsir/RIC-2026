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


# ----------------ÉTAPE 2 : MLP — Forward + ReLU + MSE uniquement-----------------


class MLP:

    def __init__(self, layer_sizes):
        """
        layer_sizes : ex [2, 64, 64, 1]
        Initialisation Xavier des poids.
        """
        self.weights = []
        self.biaises = []

        for i in range(len(layer_sizes) - 1):
            n_in  = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            std   = np.sqrt(2.0 / n_in)   # He initialization
            self.weights.append(np.random.randn(n_in, n_out) * std)
            self.biaises.append(np.zeros((1, n_out)))

    # ---- Activation ReLU ----

    def relu(self, z):
        """ReLU : remplace les valeurs négatives par 0"""
        return np.maximum(0, z)

    # ---- Passe avant (Forward) ----

    def forward(self, X):
        """
        Propage X à travers toutes les couches.
        Couches cachées  → ReLU
        Couche de sortie → linéaire (régression)
        """
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biaises)):
            z = a @ W + b                        # combinaison linéaire
            if i < len(self.weights) - 1:
                a = self.relu(z)                 # couche cachée
            else:
                a = z                            # sortie linéaire
        return a

    # ---- Fonction de perte MSE ----

    def mse(self, y_pred, y_true):
        """Mean Squared Error : moyenne de (y_pred - y_true)²"""
        return np.mean((y_pred - y_true) ** 2)



# ---------TEST----------------------------------------------------


np.random.seed(42)
mlp = MLP(layer_sizes=[2, 64, 64, 1])

# Afficher l'architecture
print("=== Architecture ===")
for i, (W, b) in enumerate(zip(mlp.weights, mlp.biaises)):
    print(f"  Couche {i+1} : W={W.shape}  b={b.shape}")

# Forward pass
y_true = z_norm.reshape(-1, 1)
y_pred = mlp.forward(X_norm)

# MSE (avant entraînement, les poids sont aléatoires)
perte = mlp.mse(y_pred, y_true)
print(f"\nMSE initiale (poids aléatoires) : {perte:.6f}")
print(f"Shape sortie : {y_pred.shape}")   # doit être (2000, 1)