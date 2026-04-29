import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARTIE 1 : Fonction cible
# ============================================================

def f(x, y):
    # f(x, y) = sin(sqrt(x^2 + y^2)) + 0.5*cos(2x + 2y)
    r = np.sqrt(x**2 + y**2)
    return np.sin(r) + 0.5 * np.cos(2*x + 2*y)


# ============================================================
# PARTIE 2 : Génération du dataset
# ============================================================

n_points = 2000
x_min, x_max = -5, 5
y_min, y_max = -5, 5

np.random.seed(42)
x = np.random.uniform(x_min, x_max, n_points)
y = np.random.uniform(y_min, y_max, n_points)
z = f(x, y)


# ============================================================
# PARTIE 3 : Normalisation
# ============================================================

X = np.column_stack([x, y])
X_mean = np.mean(X, axis=0)
X_std  = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

z_mean = np.mean(z)
z_std  = np.std(z)
z_norm = (z - z_mean) / z_std


# ============================================================
# PARTIE 4 : Affichage vérité terrain
# ============================================================

plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=10, alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Dataset - Vérité Terrain (2000 points)')
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig('C:/Users/DELL-Latitude-i7/Downloads/RIC-2026-main/RIC-2026-main/p1_verite_terrain.png', dpi=150)
plt.show()


# ============================================================
# PARTIE 5 : Architecture MLP [2 -> 64 -> 64 -> 1]
# ============================================================

class MLP:

    def __init__(self, layer_sizes, lr=5e-3, momentum=0.9):
        self.layer_sizes = layer_sizes
        self.lr          = lr
        self.momentum    = momentum
        self.n_layers    = len(layer_sizes) - 1

        # Initialisation He : std = sqrt(2 / fan_in)
        self.W = []
        self.b = []
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            std    = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(fan_in, layer_sizes[i+1]) * std)
            self.b.append(np.zeros((1, layer_sizes[i+1])))

        # Vélocités momentum
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_deriv(z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.Z = []
        self.A = [X]
        current = X
        for i in range(self.n_layers):
            z = current @ self.W[i] + self.b[i]
            self.Z.append(z)
            if i < self.n_layers - 1:
                current = self.relu(z)
            else:
                current = z
            self.A.append(current)
        return current

    @staticmethod
    def mse(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_true):
        m = y_true.shape[0]
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        delta = (2.0 / m) * (self.A[-1] - y_true)

        for i in reversed(range(self.n_layers)):
            grads_W[i] = self.A[i].T @ delta
            grads_b[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.relu_deriv(self.Z[i-1])

        return grads_W, grads_b

    def update(self, grads_W, grads_b):
        for i in range(self.n_layers):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * grads_W[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * grads_b[i]
            self.W[i]  += self.vW[i]
            self.b[i]  += self.vb[i]

    def predict(self, X):
        current = X
        for i in range(self.n_layers):
            z = current @ self.W[i] + self.b[i]
            current = self.relu(z) if i < self.n_layers - 1 else z
        return current

    def summary(self):
        print("\n" + "=" * 48)
        print("  ARCHITECTURE MLP")
        print("=" * 48)
        total = 0
        for i in range(self.n_layers):
            act    = "ReLU" if i < self.n_layers - 1 else "Lineaire"
            params = self.W[i].size + self.b[i].size
            total += params
            print(f"  Couche {i+1} : {self.layer_sizes[i]:3d} -> "
                  f"{self.layer_sizes[i+1]:3d}  |  {act:<9}  |  {params} params")
        print("-" * 48)
        print(f"  Total parametres : {total}")
        print("=" * 48 + "\n")


# ============================================================
# PARTIE 6 : Entraînement
# ============================================================

# Reshape z_norm en colonne (batch_size, 1)
z_norm_col = z_norm.reshape(-1, 1)

# Split train / validation 80% / 20%
idx   = np.random.permutation(n_points)
split = int(0.8 * n_points)

X_train = X_norm[idx[:split]]
X_val   = X_norm[idx[split:]]
z_train = z_norm_col[idx[:split]]
z_val   = z_norm_col[idx[split:]]

# Instanciation
model = MLP(layer_sizes=[2, 64, 64, 1], lr=5e-3, momentum=0.9)
model.summary()

# Hyperparamètres
epochs     = 800
batch_size = 128
n_train    = X_train.shape[0]

history_train = []
history_val   = []

print("Entraînement en cours...")
for epoch in range(1, epochs + 1):

    # Mélange aléatoire à chaque époque
    perm   = np.random.permutation(n_train)
    X_shuf = X_train[perm]
    z_shuf = z_train[perm]

    # Mini-batches
    for start in range(0, n_train, batch_size):
        Xb = X_shuf[start:start + batch_size]
        zb = z_shuf[start:start + batch_size]
        model.forward(Xb)
        gW, gb = model.backward(zb)
        model.update(gW, gb)

    # Calcul des métriques
    loss_train = model.mse(model.forward(X_train), z_train)
    loss_val   = model.mse(model.predict(X_val),   z_val)
    history_train.append(loss_train)
    history_val.append(loss_val)

    if epoch % 100 == 0:
        print(f"  Epoque {epoch:4d}/{epochs}  |  "
              f"Loss train : {loss_train:.5f}  |  "
              f"Loss val : {loss_val:.5f}")

print("\nEntraînement terminé !")


# ============================================================
# PARTIE 7 : Courbe de loss
# ============================================================

plt.figure(figsize=(9, 4))
plt.plot(history_train, label='Train MSE',      color='royalblue')
plt.plot(history_val,   label='Validation MSE', color='tomato', linestyle='--')
plt.xlabel('Époque')
plt.ylabel('MSE')
plt.title('Courbe d\'apprentissage — MLP [2, 64, 64, 1]')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('C:/Users/DELL-Latitude-i7/Downloads/RIC-2026-main/RIC-2026-main/p2_loss_curve.png', dpi=150)
plt.show()


# ============================================================
# PARTIE 8 : Comparaison MLP vs fonction originale
# ============================================================

# Grille régulière 100x100 sur [-5, 5]
grid_res = 100
gx = np.linspace(-5, 5, grid_res)
gy = np.linspace(-5, 5, grid_res)
GX, GY = np.meshgrid(gx, gy)

# Préparer les points de la grille normalisés
grid_points = np.stack([GX.ravel(), GY.ravel()], axis=1)
grid_norm   = (grid_points - X_mean) / X_std

# Prédictions MLP (dénormalisées)
z_pred_norm = model.predict(grid_norm)
z_pred      = (z_pred_norm * z_std + z_mean).reshape(grid_res, grid_res)

# Vérité terrain
Z_true = f(GX, GY)

# Affichage côte à côte
vmin = min(Z_true.min(), z_pred.min())
vmax = max(Z_true.max(), z_pred.max())
kw   = dict(extent=[-5,5,-5,5], origin='lower',
            cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im0 = axes[0].imshow(Z_true, **kw)
axes[0].set_title('f(x,y) — Vérité terrain')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(z_pred, **kw)
axes[1].set_title('MLP [2,64,64,1] — Prédiction')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(np.abs(Z_true - z_pred),
                     extent=[-5,5,-5,5], origin='lower',
                     cmap='hot', aspect='auto')
axes[2].set_title('Erreur absolue |vrai - prédit|')
plt.colorbar(im2, ax=axes[2])

for ax in axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.savefig('C:/Users/DELL-Latitude-i7/Downloads/RIC-2026-main/RIC-2026-main/p3_comparaison.png', dpi=150)
plt.show()

print(f"\nLoss finale — train : {history_train[-1]:.5f} | val : {history_val[-1]:.5f}")