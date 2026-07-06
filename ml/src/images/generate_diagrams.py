#!/usr/bin/env python3
"""Generate all bob-replacement SVG diagrams using matplotlib."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os, textwrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def setup_ax(ax, title='', xlim=(0, 10), ylim=(0, 10)):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

def box(ax, x, y, w, h, text, color='#e0e0e0', text_color='#1a1a1a', fontsize=9, ha='center', va='center', linewidth=1.5, roundness=0.05, fontweight='normal'):
    """Draw a rounded box with text."""
    from matplotlib.patches import FancyBboxPatch
    bbox = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={roundness}",
                          facecolor=color, edgecolor='#555555', linewidth=linewidth, zorder=2)
    ax.add_patch(bbox)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=text_color, zorder=3, fontweight=fontweight)

def arrow(ax, x1, y1, x2, y2, label='', color='#888888', linewidth=2, fontsize=8):
    """Draw an arrow between points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth, shrinkA=5, shrinkB=5),
                zorder=1)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.2, label, ha='center', va='bottom', fontsize=fontsize, color='#666666')

def dashed_arrow(ax, x1, y1, x2, y2, label='', color='#999999', linewidth=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth, linestyle='dashed', shrinkA=5, shrinkB=5),
                zorder=1)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.2, label, ha='center', va='bottom', fontsize=fontsize, color='#999999')

def save(name):
    plt.savefig(os.path.join(OUT, name), bbox_inches='tight', dpi=150, format='svg', transparent=True)
    plt.close()
    print(f'  {name}')

# ================================================================
# Diagram 1: L1 vs L2 Regularization
# ================================================================
fig, ax = plt.subplots(figsize=(9, 4))
setup_ax(ax)
box(ax, 2.5, 5, 4, 3.5, 'L1 (Lasso)\nLoss + λ|w|\nSparse weights\nFeature selection', color='#e8f4f8')
box(ax, 7.5, 5, 4, 3.5, 'L2 (Ridge)\nLoss + λw²\nShrinks weights\nNo sparsity', color='#fef9e7')
ax.text(5, 8.5, 'Correlated features: L1 picks one, L2 spreads evenly', ha='center', fontsize=9, fontstyle='italic', color='#555')
ax.text(5, 0.5, 'Elastic Net = L1 + L2 hybrid (best of both)', ha='center', fontsize=10, fontweight='bold', color='#2c3e50')
save('bob_L1_L2_regularization.svg')

# ================================================================
# Diagram 2: k-fold Cross Validation
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))
setup_ax(ax)
box(ax, 4, 8.5, 3, 1.2, 'Full Dataset', fontsize=10)
for i, x in enumerate([1.5, 4, 6.5]):
    box(ax, x, 6, 2, 1, f'Fold {i+1}\nTrain+Val', fontsize=8)
    arrow(ax, 4, 8, x, 6.6)
    box(ax, x, 3.5, 2, 1, f'Model {i+1}', fontsize=8)
    arrow(ax, x, 5.5, x, 4)
arrow(ax, 1.5, 3, 4, 1.5, '', color='#aaaaaa', linewidth=0)
arrow(ax, 4, 3, 4, 1.5, '', color='#aaaaaa', linewidth=0)
arrow(ax, 6.5, 3, 4, 1.5, '', color='#aaaaaa', linewidth=0)
# connection lines
ax.plot([1.5, 6.5], [2.8, 2.8], color='#aaaaaa', linewidth=1, zorder=1)
ax.plot([4, 4], [2.8, 2.3], color='#aaaaaa', linewidth=1, zorder=1)
ax.annotate('', xy=(4, 1.5), xytext=(4, 2.3), arrowprops=dict(arrowstyle='->', color='#888888', lw=2))
box(ax, 4, 0.8, 3, 1, 'Average Score', fontsize=9)
save('bob_kfold_cv.svg')

# ================================================================
# Diagram 3: Random Forest
# ================================================================
fig, ax = plt.subplots(figsize=(8, 6))
setup_ax(ax)
box(ax, 4, 9, 3.5, 1, 'Training Data', fontsize=11, color='#d5f5e3')
for i, x in enumerate([1, 4, 7]):
    box(ax, x, 6.5, 2, 1.2, f'Bootstrap\nSample {i+1}', fontsize=7.5)
    arrow(ax, 4, 8.5, x, 7.2)
    box(ax, x, 4, 2, 1.2, f'Tree {i+1}\n(high var)', fontsize=7.5)
    arrow(ax, x, 5.9, x, 4.7)
# convergence lines
ax.plot([1, 7], [3.3, 3.3], color='#aaaaaa', linewidth=1, zorder=1)
ax.plot([4, 4], [3.3, 2.8], color='#aaaaaa', linewidth=1, zorder=1)
ax.annotate('', xy=(4, 1.8), xytext=(4, 2.8), arrowprops=dict(arrowstyle='->', color='#888888', lw=2))
box(ax, 4, 1, 4, 1.5, 'AVERAGE / VOTE\n(reduces variance)', fontsize=9, color='#d5f5e3')
save('bob_random_forest.svg')

# ================================================================
# Diagram 4: Bagging vs Boosting
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for ax, title, color in [(ax1, 'Bagging', '#d5f5e3'), (ax2, 'Boosting', '#fdebd0')]:
    setup_ax(ax, title=title, xlim=(0, 10), ylim=(0, 10))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    box(ax, 5, 9, 3, 0.8, 'Data', fontsize=9, color=color)
box(ax1, 5, 6.5, 3, 1, 'Bootstrap 1\nBootstrap 2\n...k times', fontsize=7.5, color=color)
arrow(ax1, 5, 8.5, 5, 7)
box(ax1, 5, 3.5, 3, 1.5, 'Model 1\nModel 2\n(parallel)', fontsize=7.5, color=color)
arrow(ax1, 5, 6, 5, 4.5)
arrow(ax1, 5, 3, 5, 2.2)
box(ax1, 5, 0.8, 3.5, 1.2, 'AVERAGE\n↓ Variance', fontsize=8, color='#a9dfbf')

box(ax2, 5, 6.5, 3, 0.8, 'Model 1\n(weak)', fontsize=7.5, color=color)
arrow(ax2, 5, 8.5, 5, 7)
box(ax2, 5, 4.5, 3, 0.8, 'Residuals\n/ Errors', fontsize=7.5, color=color)
arrow(ax2, 5, 5.7, 5, 5.3)
box(ax2, 5, 2.5, 3, 0.8, 'Model 2\n(corrects)', fontsize=7.5, color=color)
arrow(ax2, 5, 4, 5, 3.3)
arrow(ax2, 5, 2, 5, 1.2)
box(ax2, 5, 0.3, 3.5, 1, 'SEQUENTIAL\n↓ Bias', fontsize=8, color='#f5cba7')
save('bob_bagging_boosting.svg')

# ================================================================
# Diagram 5: SVM
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
setup_ax(ax1, title='Binary SVM', xlim=(0, 10), ylim=(0, 10))
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
box(ax1, 5, 8.5, 3, 0.8, 'Input Space', fontsize=9)
arrow(ax1, 5, 8, 5, 7.2, 'Kernel Trick')
box(ax1, 5, 6, 3.5, 0.8, 'Higher-Dim\nFeature Space', fontsize=8)
arrow(ax1, 5, 5.5, 5, 4.7)
box(ax1, 5, 3.5, 3, 0.8, 'Max-Margin\nHyperplane', fontsize=8)
ax1.text(5, 2.2, 'Support Vectors\ndefine the margin', ha='center', fontsize=8, color='#555')

setup_ax(ax2, title='Multi-Class', xlim=(0, 10), ylim=(0, 10))
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
box(ax2, 5, 8.5, 3.5, 0.8, 'One-vs-One\nN(N-1)/2 clf', fontsize=8)
arrow(ax2, 5, 8, 5, 7.2)
box(ax2, 5, 6, 3, 0.8, 'Majority Vote', fontsize=9)
box(ax2, 5, 4, 3.5, 0.8, 'One-vs-Rest\nN classifiers', fontsize=8)
arrow(ax2, 5, 3.5, 5, 2.7)
box(ax2, 5, 1.5, 3, 0.8, 'Pick Highest\nConfidence', fontsize=8)
save('bob_svm.svg')

# ================================================================
# Diagram 6: CNN Architecture
# ================================================================
fig, ax = plt.subplots(figsize=(10, 3.5))
setup_ax(ax)
labels = ['Input\n32x32', 'Conv\n3x3x6\nReLU', 'Pool\n2x2\nMax', 'Conv\n3x3x16\nReLU', 'Pool\n2x2\nMax', 'Flatten\n400', 'FC\n120', 'Output\n10']
colors = ['#d5f5e3', '#e8f4f8', '#fef9e7', '#e8f4f8', '#fef9e7', '#fdebd0', '#fdebd0', '#d5f5e3']
for i, (lbl, c) in enumerate(zip(labels, colors)):
    box(ax, 1 + i*1.15, 2, 0.9, 1.8, lbl, fontsize=6.5, color=c)
    if i < 7:
        arrow(ax, 1.5 + i*1.15, 2, 1.5 + i*1.15 + 0.65, 2, '', color='#888888', linewidth=1.5)
ax.text(3, 0.3, '~~ Feature Extraction ~~', ha='center', fontsize=8, color='#999', fontstyle='italic')
ax.text(7.5, 0.3, '~~ Classification ~~', ha='center', fontsize=8, color='#999', fontstyle='italic')
save('bob_cnn.svg')

# ================================================================
# Diagram 7: RNN/LSTM
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
setup_ax(ax1, title='RNN (unrolled)', xlim=(0, 10), ylim=(0, 10))
for i, x in enumerate([1, 3.5, 6]):
    box(ax1, x, 6.5, 1.5, 0.8, f'rnn\ncell', fontsize=7, color='#e8f4f8')
    if i < 2:
        arrow(ax1, x+0.8, 6.5, x+1.5, 6.5, '', color='#888')
    ax1.text(x, 5, f'x(t{["-1","","+1"][i]})', ha='center', fontsize=7, color='#555')
    ax1.text(x, 4.2, f'h(t{["-1","","+1"][i]})', ha='center', fontsize=7, color='#555')
ax1.text(3.5, 2.5, 'Hidden state passes\ncontext across timesteps$\\rightarrow$\nVanishing gradient problem', ha='center', fontsize=8, fontstyle='italic', color='#666')
ax1.add_patch(plt.Circle((3.5, 3.5), 0.3, color='#e74c3c', alpha=0.3, zorder=1))

setup_ax(ax2, title='LSTM', xlim=(0, 10), ylim=(0, 10))
# Cell state highway
ax2.plot([1, 9], [7.5, 7.5], color='#e74c3c', linewidth=3, alpha=0.5, zorder=0)
ax2.text(5, 8.2, 'Cell state $c_t$ — gradient highway', ha='center', fontsize=8, color='#e74c3c', fontstyle='italic')
for i, (x, label) in enumerate([(1.5, 'Forget\nGate'), (5, 'Input\nGate'), (8.5, 'Output\nGate')]):
    box(ax2, x, 5, 2, 1.2, label, fontsize=7, color='#fef9e7')
    arrow(ax2, x, 6, x, 5.6, '', color='#888')
box(ax2, 5, 3, 3, 0.8, '3 Gates\nF, I, O', fontsize=7, color='#fdebd0')
ax2.text(5, 1.5, 'Additive updates $\nrightarrow$ no vanishing gradient', ha='center', fontsize=8, fontstyle='italic', color='#666')
save('bob_rnn_lstm.svg')

# ================================================================
# Diagram 8: Attention (single vs multi-head)
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
setup_ax(ax1, title='Single-Head Attention', xlim=(0, 10), ylim=(0, 10))
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
box(ax1, 5, 9, 2.5, 0.7, 'Input', fontsize=9)
arrow(ax1, 5, 8.5, 5, 7.8)
# Q, K, V as text
ax1.text(2.5, 7, 'Q Proj', ha='center', fontsize=8, color='#2c3e50',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4f8', edgecolor='#888'))
ax1.text(5, 7, 'K Proj', ha='center', fontsize=8, color='#2c3e50',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4f8', edgecolor='#888'))
ax1.text(7.5, 7, 'V Proj', ha='center', fontsize=8, color='#2c3e50',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4f8', edgecolor='#888'))
arrow(ax1, 2.5, 6.5, 5, 5.8, '', color='#888')
arrow(ax1, 7.5, 6.5, 5, 5.8, '', color='#888')
box(ax1, 5, 5, 3, 0.7, 'Q · Kᵀ / √dₖ', fontsize=8)
arrow(ax1, 5, 4.5, 5, 3.8)
box(ax1, 5, 3, 2.5, 0.7, 'Softmax', fontsize=9)
arrow(ax1, 5, 2.5, 5, 1.8)
box(ax1, 5, 1, 3, 0.7, 'Weighted Sum', fontsize=9)

setup_ax(ax2, title='Multi-Head Attention', xlim=(0, 10), ylim=(0, 10))
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
for i, x in enumerate([1.5, 4, 6.5, 9]):
    box(ax2, x, 8, 1.5, 0.7, f'Head {i+1}', fontsize=7, color='#e8f4f8')
    ax2.plot([x, 5], [7.5, 6.5], color='#888', linewidth=1, zorder=0)
box(ax2, 5, 5.5, 3, 0.7, 'Concat', fontsize=9)
arrow(ax2, 5, 5, 5, 4.3)
box(ax2, 5, 3.5, 3.5, 0.7, 'Output Projection', fontsize=8)
save('bob_attention.svg')

# ================================================================
# Diagram 9: RAG vs Fine-tuning
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for ax, title in [(ax1, 'RAG'), (ax2, 'Fine-Tuning')]:
    setup_ax(ax, title=title, xlim=(0, 10), ylim=(0, 10))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
box(ax1, 5, 8.5, 2.5, 0.7, 'Query', fontsize=9)
arrow(ax1, 5, 8, 5, 7.2)
box(ax1, 5, 6, 3, 0.8, 'Retrieve from\nKnowledge Base', fontsize=7.5, color='#e8f4f8')
arrow(ax1, 5, 5.5, 5, 4.7)
box(ax1, 5, 3.5, 3, 0.8, 'Context + Query\n→ LLM', fontsize=7.5)
arrow(ax1, 5, 3, 5, 2.2)
box(ax1, 5, 1, 3, 0.7, 'Answer + Sources', fontsize=8)

box(ax2, 5, 8.5, 3, 0.7, 'Domain Data', fontsize=9)
arrow(ax2, 5, 8, 5, 7.2)
box(ax2, 5, 6, 3, 0.8, 'Update Weights\nvia Training', fontsize=7.5, color='#fdebd0')
arrow(ax2, 5, 5.5, 5, 4.7)
box(ax2, 5, 3.5, 3, 0.8, 'Specialized\nModel', fontsize=7.5, color='#fdebd0')
box(ax2, 5, 1, 2.5, 0.7, 'Answer', fontsize=9)
ax2.text(3.7, 1.9, 'Query', ha='center', fontsize=7, color='#555')
ax2.annotate('', xy=(5, 1.8), xytext=(3.7, 2.3), arrowprops=dict(arrowstyle='->', color='#888', lw=1))
save('bob_rag_finetune.svg')

# ================================================================
# Diagram 10: Supervised vs Unsupervised taxonomy
# ================================================================
fig, ax = plt.subplots(figsize=(10, 4))
setup_ax(ax)
taxonomies = [
    (1.5, 'Supervised\nInput → Label\nLearn mapping\n$f(x)=y$', '#d5f5e3'),
    (4, 'Unsupervised\nNo labels\nFind structure\nClustering', '#e8f4f8'),
    (6.5, 'Semi-Supervised\nFew labels +\nmany unlabeled\nPseudo-labeling', '#fef9e7'),
    (9, 'Active Learning\nModel selects\nwhat to label\nMax efficiency', '#fdebd0'),
]
for x, text, color in taxonomies:
    box(ax, x, 2.5, 2.5, 2, text, fontsize=7.5, color=color)
    ax.plot([x, x], [4, 4.5], color='#888', linewidth=1, zorder=0)
ax.text(5, 4.8, 'Learning Paradigms', ha='center', fontsize=12, fontweight='bold')
save('bob_learning_paradigms.svg')

# ================================================================
# Diagram 11: Generative vs Discriminative
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
setup_ax(ax1, xlim=(0, 10), ylim=(0, 10))
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
box(ax1, 5, 6.5, 4, 1.5, 'Generative\n$P(x, y)$ = Joint Distribution\nCan GENERATE new samples', fontsize=8, color='#d5f5e3')
ax1.text(5, 3.5, 'e.g. Naive Bayes, GANs,\nDiffusion Models\n\nGood with limited labels\nCan sample from distribution', ha='center', fontsize=7.5, color='#555')

setup_ax(ax2, xlim=(0, 10), ylim=(0, 10))
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
box(ax2, 5, 6.5, 4, 1.5, 'Discriminative\n$P(y|x)$ = Conditional\nDraws decision boundary', fontsize=8, color='#e8f4f8')
ax2.text(5, 3.5, 'e.g. Logistic Regression,\nSVM, Neural Networks\n\nMore accurate with\nsufficient labeled data', ha='center', fontsize=7.5, color='#555')
save('bob_gen_disc.svg')

# ================================================================
# Diagram 12: Linear vs Logistic Regression
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
setup_ax(ax1, xlim=(0, 10), ylim=(0, 10))
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
box(ax1, 5, 8, 4, 1, 'Linear Regression', fontsize=10, fontweight='bold', color='#d5f5e3')
ax1.text(5, 5.5, '$y = w\\cdot x + b$\nContinuous output\nMSE loss\nClosed-form solution\n(Normal Equation)', ha='center', fontsize=8, color='#1a1a1a')
ax1.plot([2, 8], [1, 5], color='#3498db', linewidth=2, zorder=1)
ax1.scatter([2, 3, 4, 5, 6, 7], [1, 2.5, 2, 4, 3.5, 5], s=20, color='#e74c3c', zorder=2)
ax1.text(5, 0.5, 'Line through points', ha='center', fontsize=7, color='#666')

setup_ax(ax2, xlim=(0, 10), ylim=(0, 10))
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
box(ax2, 5, 8, 4.5, 1, 'Logistic Regression', fontsize=10, fontweight='bold', color='#e8f4f8')
ax2.text(5, 5.5, '$p = \\sigma(w\\cdot x + b)$\nProbability in $(0,1)$\nCross-entropy loss\nConvex (global optimum)', ha='center', fontsize=8, color='#1a1a1a')
x_s = np.linspace(0, 10, 100)
ax2.plot(x_s, 1/(1+np.exp(-(x_s-5)*1.5)), color='#e74c3c', linewidth=2, zorder=1)
ax2.text(5, 0.5, 'Sigmoid squashes to [0,1]', ha='center', fontsize=7, color='#666')
save('bob_linear_logistic.svg')

# ================================================================
# Diagram 13: k-means clustering
# ================================================================
fig, ax = plt.subplots(figsize=(6, 5))
setup_ax(ax)
box(ax, 3, 9, 4, 0.7, 'k-means Clustering', fontsize=11, fontweight='bold')
steps = ['1. Pick k centroids', '2. Assign points to nearest', '3. Recompute centroids', '4. Repeat 2-3 till convergence', '\nChoose k: Elbow / Silhouette']
for i, (step, y) in enumerate(zip(steps, [7.5, 6, 4.5, 3, 1.2])):
    box(ax, 3, y, 4, 0.7, step, fontsize=7.5, color='#e8f4f8' if i < 4 else '#fef9e7')
    if i < 4:
        arrow(ax, 3, y-0.4, 3, y-1.1, '', color='#888', linewidth=1)
# cluster visualization
np.random.seed(42)
for cx, cy, n, color in [(1.5, 4, 10, '#3498db'), (4, 2.5, 8, '#e74c3c'), (3, 3, 7, '#2ecc71')]:
    pts = np.random.randn(n, 2) * 0.4 + np.array([cx, cy])
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=color, alpha=0.7, zorder=2)
    ax.scatter(cx, cy, s=50, color=color, marker='x', zorder=3)
ax.set_xlim(-1, 7); ax.set_ylim(-1, 10)
save('bob_kmeans.svg')

# ================================================================
# Diagram 14: Backpropagation
# ================================================================
fig, ax = plt.subplots(figsize=(8, 3))
setup_ax(ax)
box(ax, 1, 2, 1.2, 0.7, 'x', fontsize=10)
for i, lbl in enumerate(['Layer 1', 'Layer 2', 'Loss']):
    box(ax, 3 + i*2, 2, 1.5, 0.7, lbl, fontsize=9, color='#e8f4f8')
    arrow(ax, 1.7 + i*2, 2, 2.2 + i*2, 2, '', color='#888')
# backward arrows
for i in range(3):
    ax.annotate('', xy=(1 + i*2, 1.3), xytext=(1 + i*2 + 2, 1.3),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2, linestyle='dashed'))
ax.text(4, 0.5, 'Forward (black) →  Backward (red, dashed) ←', ha='center', fontsize=9, color='#555')
ax.text(4, 0.1, 'Chain rule: $\\partial L/\\partial w = \\partial L/\\partial y \\cdot \\partial y/\\partial z \\cdot \\partial z/\\partial w$', ha='center', fontsize=8, color='#888')
save('bob_backprop.svg')

# ================================================================
# Diagram 15: Feature Store
# ================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
setup_ax(ax)
box(ax, 3, 7.5, 3.5, 0.8, 'Training Pipeline', fontsize=9, color='#d5f5e3')
box(ax, 7, 7.5, 3.5, 0.8, 'Serving Pipeline', fontsize=9, color='#fdebd0')
arrow(ax, 5, 7, 5, 5.5, '', color='#888')
arrow(ax, 8.5, 7, 8.5, 5.5, '', color='#888')
box(ax, 5, 4, 5, 1.5, 'FEATURE STORE\nFeast / Tecton\n\nOffline: batch compute\nOnline: low-latency KV\nPoint-in-time correctness', fontsize=7.5, color='#fef9e7', linewidth=2.5)
ax.plot([5, 5], [3.2, 2.5], color='#888', linewidth=1, zorder=1)
ax.text(5, 1.5, 'No training/serving skew!\nSingle source of truth for features', ha='center', fontsize=8, color='#27ae60', fontstyle='italic')
save('bob_feature_store.svg')

# ================================================================
# Diagram 16: ML Project Lifecycle
# ================================================================
fig, ax = plt.subplots(figsize=(6, 6))
setup_ax(ax)
phases = [
    (5, 8.5, 'Frame\nProblem & Metric', '#e8f4f8'),
    (8, 6, 'Data\nCollect & Validate', '#d5f5e3'),
    (8, 3.5, 'Model\nBuild & Tune', '#fef9e7'),
    (5, 1, 'Evaluate\nOffline + Online', '#fdebd0'),
    (2, 3.5, 'Deploy\nCanary → Shadow', '#e8daef'),
    (2, 6, 'Monitor\nDrift & Performance', '#fadbd8'),
]
for x, y, label, color in phases:
    box(ax, x, y, 2.5, 1.2, label, fontsize=7.5, color=color, roundness=0.08)
# arrows in a circle-like path
arrows = [(5, 7.5, 8, 6.8), (8, 5.2, 8, 4.8), (8, 2.8, 5, 2.3),
          (5, 2.3, 2, 4.3), (2, 4.8, 2, 5.2), (2, 6.8, 5, 7.5)]
for x1, y1, x2, y2 in arrows:
    arrow(ax, x1, y1, x2, y2, '', color='#888', linewidth=1.5)
ax.text(5, 9.5, 'ML Project Lifecycle', ha='center', fontsize=12, fontweight='bold')
save('bob_ml_lifecycle.svg')

# ================================================================
# Diagram 17: STAR Framework
# ================================================================
fig, ax = plt.subplots(figsize=(5, 5))
setup_ax(ax)
star = [
    (8, 'SITUATION\nWhat was the context?'),
    (6, 'TASK\nWhat was your\nresponsibility?'),
    (4, 'ACTION\nWhat did you do?\n(quantify!)'),
    (2, 'RESULT\nWhat happened?\n(metrics!)'),
]
for y, (x, label) in enumerate(star):
    box(ax, 5, x, 4, 1, label, fontsize=8, color=['#e8f4f8', '#d5f5e3', '#fef9e7', '#fdebd0'][y], roundness=0.08)
    if y < 3:
        arrow(ax, 5, x-0.5, 5, star[y+1][0]+0.5, '', color='#888', linewidth=1.5)
ax.text(5, 9.5, 'STAR Story Framework', ha='center', fontsize=12, fontweight='bold')
ax.text(5, 0.5, 'Prepare 3-4 stories for behavioral rounds', ha='center', fontsize=8, color='#666', fontstyle='italic')
save('bob_star.svg')

# ================================================================
# Diagram 18: Framing ML Problems
# ================================================================
fig, ax = plt.subplots(figsize=(9, 5))
setup_ax(ax)
box(ax, 4.5, 9, 4, 0.8, 'Business Problem\n"increase revenue"', fontsize=9, fontweight='bold', color='#fdebd0')
# clarifying questions
questions = [
    (1.5, 6.5, 'Objective\n& Metric?'),
    (3.5, 6.5, 'Scale\nReq/sec?'),
    (5.5, 6.5, 'Latency\nBudget?'),
    (7.5, 6.5, 'Data\nAvailable?'),
]
for x, y, label in questions:
    box(ax, x, y, 1.8, 1, label, fontsize=7.5, color='#e8f4f8')
    arrow(ax, x, y+0.6, x, y+1.2, '', color='#888')
arrow(ax, 1.5, 5.8, 4.5, 4.8, '', color='#888')
arrow(ax, 3.5, 5.8, 4.5, 4.8, '', color='#888')
arrow(ax, 5.5, 5.8, 4.5, 4.8, '', color='#888')
arrow(ax, 7.5, 5.8, 4.5, 4.8, '', color='#888')
box(ax, 4.5, 3.5, 3.5, 0.8, 'ML Problem:\nPredict purchase prob.', fontsize=8, color='#d5f5e3')
arrow(ax, 4.5, 3, 4.5, 2.3)
box(ax, 3, 1.3, 2.5, 0.8, 'Target +\nMetric', fontsize=8, color='#fef9e7')
box(ax, 6, 1.3, 2.5, 0.8, 'Baseline +\nSuccess', fontsize=8, color='#fef9e7')
save('bob_framing_ml.svg')

# ================================================================
# Diagram 19: Batch vs Online Inference
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
setup_ax(ax1, title='Batch Inference', xlim=(0, 10), ylim=(0, 10))
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
box(ax1, 5, 8, 3, 0.7, 'Data Lake', fontsize=9)
arrow(ax1, 5, 7.5, 5, 6.5, 'Nightly Job')
box(ax1, 5, 5, 3.5, 1, 'Precompute\nPredictions', fontsize=8, color='#e8f4f8')
arrow(ax1, 5, 4.5, 5, 3.5)
box(ax1, 5, 2, 3, 0.7, 'Key-Value\nStore', fontsize=8, color='#fef9e7')
arrow(ax1, 5, 1.5, 5, 0.8)
box(ax1, 5, 0.3, 2, 0.5, 'Serve', fontsize=8)

setup_ax(ax2, title='Online Inference', xlim=(0, 10), ylim=(0, 10))
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
box(ax2, 5, 8, 2.5, 0.7, 'Request', fontsize=9)
arrow(ax2, 5, 7.5, 5, 6.5, 'Real-time')
box(ax2, 5, 5, 3, 0.7, 'Model\n<100ms', fontsize=8, color='#fdebd0')
arrow(ax2, 5, 4.5, 5, 3.5)
box(ax2, 5, 2, 2.5, 0.7, 'Serve', fontsize=9)
save('bob_batch_online.svg')

# ================================================================
# Diagram 20: CI/CD Pipeline
# ================================================================
fig, ax = plt.subplots(figsize=(10, 3))
setup_ax(ax)
stages = [
    (1, 'Git\nPush', '#d5f5e3'),
    (2.5, 'Data\nValidate', '#e8f4f8'),
    (4, 'Train\n/Tune', '#fef9e7'),
    (5.5, 'Eval\nOffline', '#fdebd0'),
    (7, 'Canary\n1%', '#e8daef'),
    (8.5, 'Shadow\n50%', '#fadbd8'),
    (10, 'Full\nRollout', '#d5f5e3'),
]
for x, label, color in stages:
    box(ax, x, 2, 1.2, 1, label, fontsize=7, color=color)
    if stages.index((x, label, color)) < 6:
        arrow(ax, x+0.7, 2, x+1.3, 2, '', color='#888', linewidth=1.5)
# feedback loop
ax.annotate('', xy=(5.5, 0.5), xytext=(1, 0.5),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5, linestyle='dashed'))
ax.text(3, 0.3, 'Fail → re-iterate', ha='center', fontsize=7, color='#e74c3c', fontstyle='italic')
ax.annotate('', xy=(10, 0.5), xytext=(8.5, 0.5),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5, linestyle='dashed'))
ax.text(9.25, 0.3, 'Degraded → rollback', ha='center', fontsize=7, color='#e74c3c', fontstyle='italic')
save('bob_cicd.svg')

# ================================================================
# Diagram 21: Monitoring Pillars
# ================================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
setup_ax(ax)
pillars = [
    (2, 7, 'Model\nPerformance\nAccuracy, AUC', '#e8f4f8'),
    (5, 7, 'Data Quality\nMissing, PSI\nOutliers', '#d5f5e3'),
    (8, 7, 'System Health\nLatency, Throughput\nErrors', '#fef9e7'),
]
for x, y, label, color in pillars:
    box(ax, x, y, 2.2, 1.3, label, fontsize=7.5, color=color)
    arrow(ax, x, 6.2, x, 5.2, '', color='#888')
box(ax, 5, 3.5, 5.5, 1.2, 'Dashboard\nPrometheus + Grafana', fontsize=9, color='#fdebd0')
arrow(ax, 5, 2.8, 5, 1.8)
box(ax, 5, 0.5, 3.5, 1, 'Alert + Auto-Rollback', fontsize=8, color='#fadbd8')
ax.text(5, 9, 'Monitoring Pillars', ha='center', fontsize=12, fontweight='bold')
save('bob_monitoring.svg')

# ================================================================
# Diagram 22: Data vs Model Parallelism
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
setup_ax(ax1, title='Data Parallelism', xlim=(0, 10), ylim=(0, 10))
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
box(ax1, 5, 8.5, 2.5, 0.7, 'Batch', fontsize=9)
ax1.plot([2.5, 7.5], [7.8, 7.8], color='#888', linewidth=1, zorder=0)
ax1.plot([2.5, 2.5], [7.8, 7], color='#888', linewidth=1, zorder=0)
ax1.plot([7.5, 7.5], [7.8, 7], color='#888', linewidth=1, zorder=0)
for i, x in enumerate([2.5, 5, 7.5]):
    box(ax1, x, 5.5, 1.8, 0.8, f'Shard {i+1}', fontsize=7.5, color='#e8f4f8')
    arrow(ax1, x, 6.5, x, 6.2, '', color='#888')
    box(ax1, x, 3.5, 1.8, 0.8, f'Model\nReplica {i+1}', fontsize=7, color='#d5f5e3')
    arrow(ax1, x, 4.5, x, 4.2, '', color='#888')
ax1.plot([2.5, 7.5], [2.8, 2.8], color='#888', linewidth=1, zorder=0)
ax1.plot([5, 5], [2.8, 2.3], color='#888', linewidth=1, zorder=0)
arrow(ax1, 5, 1.5, 5, 2.3, '', color='#888')
box(ax1, 5, 0.5, 2.5, 0.7, 'Sync Gradients', fontsize=8)

setup_ax(ax2, title='Model / Pipeline Parallelism', xlim=(0, 10), ylim=(0, 10))
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
for i, (x, label) in enumerate([(2, 'Layer 1-3\nGPU 1'), (5, 'Layer 4-6\nGPU 2'), (8, 'Layer 7-9\nGPU 3')]):
    box(ax2, x, 4, 2, 1.5, label, fontsize=8, color='#fef9e7')
    if i < 2:
        arrow(ax2, x+1.2, 4, x+2, 4, '', color='#888')
ax2.text(5, 1, 'Pipeline parallelism splits model stages\nacross devices. Tensor parallelism\nsplits individual operations.', ha='center', fontsize=7.5, color='#666')
save('bob_parallelism.svg')

# ================================================================
# Diagram 23: Recommendation System
# ================================================================
fig, ax = plt.subplots(figsize=(8, 6))
setup_ax(ax)
box(ax, 3, 9, 2, 0.7, 'User', fontsize=9)
box(ax, 7, 9, 2, 0.7, 'Item', fontsize=9)
arrow(ax, 3, 8.5, 3, 7.5, '', color='#888')
arrow(ax, 7, 8.5, 7, 7.5, '', color='#888')
box(ax, 3, 6.5, 2, 0.7, 'User\nTower', fontsize=8, color='#e8f4f8')
box(ax, 7, 6.5, 2, 0.7, 'Item\nTower', fontsize=8, color='#e8f4f8')
ax.plot([3, 7], [6, 6], color='#888', linewidth=1, zorder=0)
ax.plot([5, 5], [6, 5.2], color='#888', linewidth=1, zorder=0)
arrow(ax, 5, 4.5, 5, 5.2, '', color='#888')
box(ax, 5, 3.5, 3, 0.7, 'Dot Product\nSimilarity', fontsize=8)
arrow(ax, 5, 3, 5, 2.2)
box(ax, 5, 1.3, 3, 0.7, 'ANN Retrieval\nTop-N', fontsize=8, color='#fef9e7')
arrow(ax, 5, 0.8, 5, 0.2)
box(ax, 5, -0.5, 2.5, 0.5, 'Serve', fontsize=8)
save('bob_recommendation.svg')

print("All 23 diagram SVGs generated!")
