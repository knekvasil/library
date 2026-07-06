"""Generate SVG diagrams for General ML Review article."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# Global: transparent background
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.transparent': True,
})

def save(name):
    plt.savefig(os.path.join(OUT, name), bbox_inches='tight', dpi=150, format='svg',
                transparent=True)
    plt.close()
    print(f'  Generated {name}')

# === 1. Bias-Variance Tradeoff ===
fig, ax = plt.subplots(figsize=(6, 4))
complexity = np.linspace(0, 1, 100)
bias = (1 - complexity)**2
variance = complexity**2
total = bias + variance + 0.05
ax.plot(complexity, bias, label='Bias', color='#e74c3c', linewidth=2)
ax.plot(complexity, variance, label='Variance', color='#3498db', linewidth=2)
ax.plot(complexity, total, label='Total error', color='#2c3e50', linewidth=2.5, linestyle='--')
ax.axvline(0.45, color='gray', alpha=0.3, linestyle=':')
ax.text(0.45, 0.95, 'Sweet spot', ha='center', fontsize=9, color='gray', fontstyle='italic')
ax.set_xlabel('Model complexity', fontsize=11)
ax.set_ylabel('Error', fontsize=11)
ax.set_title('Bias-Variance Tradeoff', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_xlim(-0.02, 1.02)
save('bias_variance_tradeoff.svg')

# === 2. Overfitting / Underfitting Loss Curves ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
np.random.seed(42)

epochs = np.linspace(1, 50, 50)
# Underfitting: both high
train_uf = 1.2 - 0.3 * np.log(epochs) + 0.05 * np.random.randn(50)
val_uf = 1.3 - 0.25 * np.log(epochs) + 0.08 * np.random.randn(50)
ax1.plot(epochs, np.maximum(train_uf, 0.1), label='Training loss', color='#3498db')
ax1.plot(epochs, np.maximum(val_uf, 0.1), label='Validation loss', color='#e74c3c')
ax1.set_title('Underfitting', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend(fontsize=8)
ax1.set_ylim(0, 1.5)
ax1.set_xlim(1, 50)

# Overfitting: train drops, val starts similar then rises
train_of = 1.2 - 0.6 * np.log(epochs) + 0.03 * np.random.randn(50)
val_of = 1.2 - 0.15 * np.log(epochs) + 0.004 * (np.maximum(epochs - 10, 0))**1.5 + 0.05 * np.random.randn(50)
# Ensure val starts at similar level to train
val_of[:5] = train_of[:5] + 0.05 + 0.02 * np.random.randn(5)
ax2.plot(epochs, np.maximum(train_of, 0.05), label='Training loss', color='#3498db')
ax2.plot(epochs, np.maximum(val_of, 0.05), label='Validation loss', color='#e74c3c')
ax2.axvline(18, color='gray', alpha=0.3, linestyle=':')
ax2.text(18, 0.15, 'Early stopping', ha='center', fontsize=8, color='gray', fontstyle='italic')
ax2.set_title('Overfitting', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epochs')
ax2.legend(fontsize=8)
ax2.set_ylim(0, 1.5)
ax2.set_xlim(1, 50)

fig.suptitle('Training vs Validation Loss', fontsize=13, fontweight='bold')
plt.tight_layout()
save('overfitting_curves.svg')

# === 3. Confusion Matrix ===
fig, ax = plt.subplots(figsize=(5, 4))
cm = np.array([[85, 10], [5, 100]])
labels = [['True Positive\n(TP)', 'False Negative\n(FN)'], ['False Positive\n(FP)', 'True Negative\n(TN)']]
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=np.max(cm) * 1.2)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}\n{labels[i][j]}', ha='center', va='center', fontsize=10, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted\nPositive', 'Predicted\nNegative'], fontsize=9)
ax.set_yticklabels(['Actual\nPositive', 'Actual\nNegative'], fontsize=9)
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.8)
save('confusion_matrix.svg')

# === 4. ROC Curve (more gradual) ===
fig, ax = plt.subplots(figsize=(5, 5))
np.random.seed(42)
n = 200
y_true = np.concatenate([np.ones(100), np.zeros(100)])
# Use Beta distributions that give a more gradual ROC curve
y_score = np.concatenate([np.random.beta(3, 1.5, 100), np.random.beta(1.5, 3, 100)])
thresholds = np.linspace(1, 0, 200)
tpr = []
fpr = []
for t in thresholds:
    tp = np.sum((y_score >= t) & (y_true == 1))
    fn = np.sum((y_score < t) & (y_true == 1))
    fp = np.sum((y_score >= t) & (y_true == 0))
    tn = np.sum((y_score < t) & (y_true == 0))
    tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
ax.plot(fpr, tpr, color='#3498db', linewidth=2, label='ROC curve (AUC ≈ 0.82)')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5, label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=10)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=10)
ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
save('roc_curve.svg')

# === 5. Gradient Descent Variants (contour) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 + 0.5 * X * Y + 0.1 * X**3

# Left: contour with gradient path
levels = np.linspace(Z.min(), Z.max(), 12)
ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
# Simulated paths
path_sgd = np.array([[-2.8, 2.5], [-2.0, 1.8], [-0.5, 0.8], [0.3, -0.2], [0.1, -0.1], [0.0, 0.0]])
path_batch = np.array([[-2.5, 2.0], [-1.2, 1.0], [-0.3, 0.2], [0.0, 0.0]])
ax1.plot(path_sgd[:, 0], path_sgd[:, 1], 'o-', color='#e74c3c', markersize=3, linewidth=1.5, label='SGD (noisy)')
ax1.plot(path_batch[:, 0], path_batch[:, 1], 's-', color='#3498db', markersize=4, linewidth=2, label='Batch GD')
ax1.scatter([0], [0], color='#2ecc71', s=100, marker='*', zorder=5, label='Minimum')
ax1.set_title('GD Paths on Loss Surface', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.set_xlabel('$w_1$')
ax1.set_ylabel('$w_2$')

# Right: convergence comparison
epochs = np.arange(1, 31)
loss_batch = 5 * np.exp(-0.15 * epochs) + 0.05
loss_sgd = 5 * np.exp(-0.08 * epochs) + 0.1 + 0.15 * np.random.randn(30)
loss_mini = 5 * np.exp(-0.12 * epochs) + 0.05 + 0.05 * np.random.randn(30)
loss_mini = np.maximum(loss_mini, 0.02)
ax2.plot(epochs, loss_batch, label='Batch GD', color='#3498db', linewidth=2)
ax2.plot(epochs, loss_sgd, label='SGD', color='#e74c3c', linewidth=1.5, alpha=0.7)
ax2.plot(epochs, loss_mini, label='Mini-batch', color='#2ecc71', linewidth=2)
ax2.set_title('Convergence Rate', fontsize=11, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(fontsize=8)

fig.suptitle('Gradient Descent Variants', fontsize=13, fontweight='bold')
plt.tight_layout()
save('gradient_descent.svg')

# === 6. PCA 2D Projection ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

np.random.seed(123)
n = 100
mean = [2, 3]
cov = [[3, 2.5], [2.5, 4]]
data = np.random.multivariate_normal(mean, cov, n)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

ax1.scatter(data[:, 0], data[:, 1], alpha=0.6, s=20, color='#3498db')
for i, (vec, val) in enumerate(zip(eigvecs.T, eigvals[idx])):
    vec_scaled = vec * np.sqrt(val) * 1.5
    ax1.arrow(mean[0], mean[1], vec_scaled[0], vec_scaled[1],
              head_width=0.2, head_length=0.3, fc='#e74c3c' if i == 0 else '#2ecc71',
              ec='#e74c3c' if i == 0 else '#2ecc71', linewidth=2.5)
ax1.set_title('Original Data with Principal Components', fontsize=10, fontweight='bold')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_aspect('equal')

# Projected
projected = data @ eigvecs.T
ax2.scatter(projected[:, 0], projected[:, 1], alpha=0.6, s=20, color='#2ecc71')
ax2.set_xlabel('PC 1 (%.1f%% variance)' % (eigvals[idx][0] / eigvals[idx].sum() * 100))
ax2.set_ylabel('PC 2 (%.1f%% variance)' % (eigvals[idx][1] / eigvals[idx].sum() * 100))
ax2.set_title('Projected onto PC1-PC2', fontsize=10, fontweight='bold')
ax2.set_aspect('equal')

fig.suptitle('Principal Component Analysis (PCA)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('pca_projection.svg')

# === 7. Gradient Boosting (sequential residuals) ===
fig, axes = plt.subplots(1, 4, figsize=(12, 3))

x_gb = np.linspace(0, 10, 100)
y_true = 2 + 0.8 * x_gb + 0.3 * np.sin(x_gb * 0.8) * x_gb ** 0.5

predictions = np.zeros_like(x_gb)
for i in range(4):
    residual = y_true - predictions
    idx = np.argmin(np.abs(x_gb[:, None] - x_gb[None, :]) * np.abs(residual[None, :]), axis=1)
    pred = residual[idx] * 0.5
    predictions += pred
    axes[i].scatter(x_gb, y_true, alpha=0.3, s=8, color='gray', label='True')
    axes[i].plot(x_gb, predictions, color='#3498db', linewidth=2)
    axes[i].set_title(f'Tree {i+1}', fontsize=9, fontweight='bold')
    axes[i].set_ylim(y_true.min() - 1, y_true.max() + 1)
    if i == 0:
        axes[i].set_ylabel('Prediction')

fig.suptitle('Gradient Boosting — Sequential Residual Fitting', fontsize=13, fontweight='bold')
plt.tight_layout()
save('gradient_boosting.svg')

# === 8. Data Drift vs Concept Drift ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

np.random.seed(42)
# Data drift: distribution shifts
t = np.linspace(1, 50, 50)
mean_t = 5 + 0.08 * t
data_drift = mean_t + np.random.randn(50) * 1.5
ax1.plot(t, data_drift, 'o-', color='#3498db', markersize=4, linewidth=1)
ax1.axvline(25, color='red', alpha=0.3, linestyle=':')
# Move label lower so it doesn't intersect title
ax1.text(25, data_drift.min() - 1, "Shift →", ha='center', color='red', fontsize=9, va='top')
ax1.set_title('Data Drift (Covariate Shift)\n$P(x)$ changes', fontsize=10, fontweight='bold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Feature Value')

# Concept drift: decision boundary shifts
t = np.linspace(1, 50, 50)
p_class1 = 1 / (1 + np.exp(-(-3 + 0.1 * t)))  # boundary shifts over time
ax2.plot(t, p_class1, color='#e74c3c', linewidth=2)
ax2.axhline(0.5, color='gray', alpha=0.3, linestyle='--')
ax2.axvline(30, color='red', alpha=0.3, linestyle=':')
# Move label lower
ax2.text(30, 0.15, 'Boundary\nshift', ha='center', color='red', fontsize=9, va='top')
ax2.set_title('Concept Drift\n$P(y|x)$ changes', fontsize=10, fontweight='bold')
ax2.set_xlabel('Time')
ax2.set_ylabel('P(class = 1)')

fig.suptitle('Data Drift vs Concept Drift', fontsize=13, fontweight='bold')
plt.tight_layout()
save('drift_types.svg')

print('All diagrams generated.')
