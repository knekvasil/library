#!/usr/bin/env python3
"""Generate all TikZ SVG diagrams for the interview prep article."""
import subprocess, os, tempfile, shutil

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)

TEX_PREAMBLE = r'''
\documentclass[tikz,border={2pt 2pt 2pt 2pt}]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usetikzlibrary{shapes,arrows,positioning,fit,backgrounds,calc,decorations.pathmorphing,patterns}
\definecolor{fgcolor}{HTML}{1a1a1a}
\definecolor{border}{HTML}{555555}
\definecolor{boxblue}{HTML}{e8f4f8}
\definecolor{boxgreen}{HTML}{d5f5e3}
\definecolor{boxyellow}{HTML}{fef9e7}
\definecolor{boxorange}{HTML}{fdebd0}
\definecolor{boxpurple}{HTML}{e8daef}
\definecolor{boxpink}{HTML}{fadbd8}
\tikzset{
  box/.style={rectangle, draw=border, rounded corners=2pt, align=center, inner sep=4pt, minimum width=2.2cm, text=fgcolor, font=\small},
  boxblue/.style={box, fill=boxblue},
  boxgreen/.style={box, fill=boxgreen},
  boxyellow/.style={box, fill=boxyellow},
  boxorange/.style={box, fill=boxorange},
  boxpurple/.style={box, fill=boxpurple},
  boxpink/.style={box, fill=boxpink},
  arr/.style={->, >=stealth, draw=fgcolor!60, line width=0.6pt},
  darr/.style={->, >=stealth, draw=fgcolor!60, line width=0.6pt, dashed},
  lbl/.style={text=fgcolor!50, font=\tiny, align=center},
  title/.style={text=fgcolor, font=\small\bfseries, align=center},
}
'''

def make_tex(filename, tikz_code):
    if not filename.endswith('.tex'):
        filename += '.tex'
    tex = TEX_PREAMBLE + r'\begin{document}'
    tex += r'\pagecolor{green!0}'  # transparent background
    tex += r'\begin{tikzpicture}'
    tex += tikz_code
    tex += r'\end{tikzpicture}'
    tex += r'\end{document}'
    path = os.path.join(OUT, filename)
    with open(path, 'w') as f:
        f.write(tex)
    return path

def compile(tex_path):
    base = os.path.splitext(tex_path)[0]
    dvi = base + '.dvi'
    svg = base + '.svg'
    # lualatex → dvi
    r = subprocess.run(
        ['lualatex', '--output-format=dvi', '--interaction=nonstopmode', tex_path],
        capture_output=True, text=True, cwd=os.path.dirname(tex_path))
    if r.returncode != 0:
        for line in r.stderr.split('\n')[-5:]:
            if 'error' in line.lower():
                print(f'  ! {line}')
    # dvisvgm → svg
    if os.path.exists(dvi):
        r2 = subprocess.run(
            ['dvisvgm', '--no-fonts', '--exact-bbox', dvi],
            capture_output=True, text=True, cwd=os.path.dirname(tex_path))
        if r2.returncode != 0 or not os.path.exists(svg):
            print(f'  dvisvgm failed for: {os.path.basename(tex_path)}')
    # Clean up aux files
    for ext in ['.aux', '.log', '.dvi', '.tex']:
        p = base + ext
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(svg):
        print(f'  {os.path.basename(svg)}')

# ================================================================
# Diagram 1: L1 vs L2 Regularization
# ================================================================
t1 = r'''
  \node[boxblue] (l1) at (0,0) {\textbf{L1 (Lasso)}\\\ \\[-2pt] $Loss + \lambda|w|$\\ Sparse weights\\ Feature selection};
  \node[boxgreen] (l2) at (5,0) {\textbf{L2 (Ridge)}\\\ \\[-2pt] $Loss + \lambda w^2$\\ Shrinks weights\\ No sparsity};
  \node[lbl] at (2.5,-1.8) {Correlated: L1 picks one, L2 spreads evenly};
  \node[title] at (2.5,1.5) {Elastic Net = L1 + L2 hybrid};
'''
make_tex('bob_L1_L2_regularization', t1)

# ================================================================
# Diagram 2: k-fold CV
# ================================================================
t2 = r'''
  \node[box,fill=boxgreen!60,minimum width=3cm] (data) at (0,3) {\small\textbf{Full Dataset}};
  \foreach \x/\l in {-2.5/Fold 1,0/Fold 2,2.5/Fold 3} {
    \node[boxblue,minimum width=1.8cm] (f\x) at (\x,1.2) {$\l$\\[-2pt] Train+Val};
    \draw[arr] (data) -- (f\x);
    \node[boxgreen,minimum width=1.8cm] (m\x) at (\x,-0.8) {Model};
    \draw[arr] (f\x) -- (m\x);
  }
  \draw[fgcolor!40] (-2.5,-1.3) -- (2.5,-1.3);
  \draw[arr] (0,-1.3) -- (0,-2);
  \node[boxyellow,minimum width=2.5cm] (avg) at (0,-2.8) {\small Average Score};
'''
make_tex('bob_kfold_cv', t2)

# ================================================================
# Diagram 3: Random Forest
# ================================================================
t3 = r'''
  \node[boxgreen,minimum width=3cm] (data) at (0,3.5) {\small\textbf{Training Data}};
  \foreach \x/\l in {-2.5/Bootstrap 1,0/Bootstrap 2,2.5/Bootstrap k} {
    \node[boxblue,minimum width=1.8cm] (b\x) at (\x,1.5) {\footnotesize $\l$};
    \draw[arr] (data) -- (b\x);
    \node[boxyellow,minimum width=1.8cm] (t\x) at (\x,-0.5) {\footnotesize Tree\\[-2pt] \footnotesize(high var)};
    \draw[arr] (b\x) -- (t\x);
  }
  \draw[fgcolor!40] (-2.5,-1) -- (2.5,-1);
  \draw[arr] (0,-1) -- (0,-1.8);
  \node[boxgreen,minimum width=3cm] (avg) at (0,-2.6) {\small\textbf{Average / Vote}\\\ \
[-2pt] \footnotesize(reduces variance)};
'''
make_tex('bob_random_forest', t3)

# ================================================================
# Diagram 4: Bagging vs Boosting
# ================================================================
t4 = r'''
  \node[title] at (-2.5,3) {BAGGING};
  \node[boxgreen,minimum width=2cm] (d1) at (-2.5,1.8) {\footnotesize Data};
  \node[boxblue,minimum width=2cm] (s1) at (-2.5,0.3) {\footnotesize Bootstrap 1\\ Bootstrap 2 ...};
  \draw[arr] (d1) -- (s1);
  \node[boxorange,minimum width=2cm] (m1) at (-2.5,-1.5) {\footnotesize Model 1\\ Model 2\\ (parallel)};
  \draw[arr] (s1) -- (m1);
  \node[boxgreen,minimum width=2.2cm] (ens1) at (-2.5,-3.2) {\scriptsize\textbf{AVERAGE}\\\ \scriptsize$\downarrow$ Variance};

  \node[title] at (2.5,3) {BOOSTING};
  \node[boxgreen,minimum width=2cm] (d2) at (2.5,1.8) {\footnotesize Data};
  \node[boxblue,minimum width=2cm] (m2) at (2.5,0.3) {\footnotesize Model 1\\ (weak)};
  \draw[arr] (d2) -- (m2);
  \node[boxyellow,minimum width=2cm] (r2) at (2.5,-1) {\footnotesize Residuals\\ Errors};
  \draw[arr] (m2) -- (r2);
  \node[boxorange,minimum width=2cm] (m3) at (2.5,-2.5) {\footnotesize Model 2\\ (corrects)};
  \draw[arr] (r2) -- (m3);
  \node[boxgreen,minimum width=2.2cm] (ens2) at (2.5,-3.8) {\scriptsize\textbf{SEQUENTIAL}\\\ \scriptsize$\downarrow$ Bias};
'''
make_tex('bob_bagging_boosting', t4)

# ================================================================
# Diagram 5: SVM
# ================================================================
t5 = r'''
  \node[title] at (-2.5,3) {Binary SVM};
  \node[boxblue] (in) at (-2.5,1.5) {\footnotesize Input\\[-2pt] \footnotesize Space};
  \node[boxgreen] (hd) at (-2.5,-0.5) {\footnotesize Higher-Dim\\[-2pt] \footnotesize Feature Space};
  \draw[arr] (in) -- (hd) node[midway,right,lbl] {Kernel};
  \node[boxyellow] (svm) at (-2.5,-2.5) {\footnotesize Max-Margin\\[-2pt] \footnotesize Hyperplane};
  \draw[arr] (hd) -- (svm);
  \node[lbl] at (-2.5,-3.8) {Support Vectors\\[-2pt] define the margin};

  \node[title] at (2.5,3) {Multi-Class};
  \node[boxblue,minimum width=2.2cm] (ovo) at (2.5,1.5) {\footnotesize One-vs-One\\[-2pt] \footnotesize N(N-1)/2};
  \node[boxgreen] (vote) at (2.5,0) {\footnotesize Majority Vote};
  \draw[arr] (ovo) -- (vote);
  \node[boxorange,minimum width=2.2cm] (ovr) at (2.5,-1.5) {\footnotesize One-vs-Rest\\[-2pt] \footnotesize N classifiers};
  \node[boxyellow] (pick) at (2.5,-3) {\footnotesize Pick Highest\\[-2pt] \footnotesize Confidence};
  \draw[arr] (ovr) -- (pick);
'''
make_tex('bob_svm', t5)

# ================================================================
# Diagram 6: CNN Architecture
# ================================================================
t6 = r'''
  \foreach \x/\l/\c in {
    0/Input\\32x32/boxgreen,
    1.2/Conv\\3x3x6/boxblue,
    2.4/Pool\\2x2/boxyellow,
    3.6/Conv\\3x3x16/boxblue,
    4.8/Pool\\2x2/boxyellow,
    6/Flatten\\400/boxorange,
    7.2/FC\\120/boxorange,
    8.4/Output\\10/boxgreen
  } {
    \node[\c,minimum width=0.9cm,minimum height=1.5cm,font=\tiny] at (\x,0) {$\l$};
  }
  \foreach \x in {0.6,1.8,3,4.2,5.4,6.6,7.8} {
    \draw[arr] (\x,0) -- (\x+0.6,0);
  }
  \node[lbl] at (3,-1.3) {Feature Extraction};
  \node[lbl] at (7.5,-1.3) {Classification};
'''
make_tex('bob_cnn', t6)

# ================================================================
# Diagram 7: RNN / LSTM
# ================================================================
t7 = r'''
  \draw[dashed,fgcolor!20] (-0.5,-1.5) rectangle (3.5,2) node[lbl,above] {RNN (unrolled)};
  \foreach \x/\i in {0.5/-1,2/0,3.5/+1} {
    \node[boxblue,minimum width=0.9cm,minimum height=0.6cm,font=\tiny] at (\x,0.8) {cell};
    \node[lbl,font=\tiny] at (\x,-0.3) {$x_{t\i}$};
    \node[lbl,font=\tiny] at (\x,-1) {$h_{t\i}$};
  }
  \draw[arr] (1,-0.3) -- (2,-0.3);
  \draw[arr] (1,0.8) -- (2,0.8);
  \draw[arr] (2,0.8) -- (3.5,0.8);

  \draw[dashed,fgcolor!20] (4.5,-1.5) rectangle (9.5,2) node[lbl,above] {LSTM};
  \draw[line width=3,fgcolor!30] (5,1.5) -- (9,1.5) node[midway,above,lbl] {Cell state $c_t$};
  \foreach \x/\l in {5.5/Forget\\Gate,7/Input\\Gate,8.5/Output\\Gate} {
    \node[boxyellow,minimum width=1.2cm,minimum height=0.8cm,font=\tiny] at (\x,0) {$\l$};
  }
  \node[boxorange,minimum width=1.5cm,font=\tiny] at (7,-1) {3 Gates F,I,O};
  \node[lbl,font=\tiny] at (7,-2.2) {Additive $\rightarrow$ no vanish};
'''
make_tex('bob_rnn_lstm', t7)

# ================================================================
# Diagram 8: Attention
# ================================================================
t8 = r'''
  \draw[dashed,fgcolor!20] (-4.5,-1.5) rectangle (0,2) node[lbl,above] {Single-Head};
  \node[boxgreen,minimum width=1.5cm] (in) at (-2.5,1.2) {Input};
  \node[boxblue] (q) at (-4,0) {Q Proj};
  \node[boxblue] (k) at (-2.5,0) {K Proj};
  \node[boxblue] (v) at (-1,0) {V Proj};
  \draw[arr] (in) -- (q); \draw[arr] (in) -- (k); \draw[arr] (in) -- (v);
  \node[boxyellow,minimum width=1.8cm] (s) at (-2.5,-1.2) {$Q\cdot K^T/\sqrt{d_k}$};
  \draw[arr] (q) -- (s); \draw[arr] (k) -- (s);
  \node[boxorange] (sm) at (-2.5,-2.2) {Softmax};
  \draw[arr] (s) -- (sm);
  \node[boxgreen] (ws) at (-2.5,-3.2) {Weighted Sum};
  \draw[arr] (sm) -- (ws);

  \draw[dashed,fgcolor!20] (1,-1.5) rectangle (5.5,2) node[lbl,above] {Multi-Head};
  \foreach \x/\l in {2/Head 1,3/Head 2,4/Head 3,5/Head 4} {
    \node[boxblue,minimum width=0.8cm,minimum height=0.6cm,font=\tiny] at (\x,0.8) {$\l$};
  }
  \node[boxorange] (cc) at (3.5,-0.8) {Concat};
  \node[boxgreen] (op) at (3.5,-2.2) {Output Proj};
  \draw[arr] (cc) -- (op);
'''
make_tex('bob_attention', t8)

# ================================================================
# Diagram 9: RAG vs Fine-tuning
# ================================================================
t9 = r'''
  \node[title] at (-2,2.5) {RAG};
  \node[boxblue] (q) at (-2,1) {Query};
  \node[boxgreen,minimum width=2cm] (r) at (-2,-0.5) {Retrieve from\\ Knowledge Base};
  \draw[arr] (q) -- (r);
  \node[boxyellow] (c) at (-2,-2) {Context + Query\\ $\rightarrow$ LLM};
  \draw[arr] (r) -- (c);
  \node[boxorange] (a) at (-2,-3.2) {Answer + Sources};
  \draw[arr] (c) -- (a);

  \node[title] at (2.5,2.5) {Fine-Tuning};
  \node[boxblue] (d) at (2.5,1) {Domain Data};
  \node[boxgreen,minimum width=2cm] (ft) at (2.5,-0.5) {Update Weights\\ via Training};
  \draw[arr] (d) -- (ft);
  \node[boxyellow] (m) at (2.5,-2) {Specialized\\ Model};
  \draw[arr] (ft) -- (m);
  \node[boxorange] (a2) at (2.5,-3.2) {Answer};
  \draw[arr] (m) -- (a2);
  \node[lbl] at (-2.5,-4) {Can combine: RAG for freshness + Fine-tune for style};
'''
make_tex('bob_rag_finetune', t9)

# ================================================================
# Diagram 10: Learning Paradigms
# ================================================================
t10 = r'''
  \node[title] at (0,2) {Learning Paradigms};
  \foreach \x/\l/\c in {
    -3.5/Supervised\\Input$\rightarrow$Label/boxblue,
    -1.2/Unsupervised\\No labels/boxgreen,
    1.2/Semi-Supervised\\Few+unlabeled/boxyellow,
    3.5/Active Learning\\Model queries/boxorange
  } {
    \node[\c,minimum width=2cm,minimum height=1.5cm,font=\footnotesize] at (\x,-0.8) {$\l$};
  }
'''
make_tex('bob_learning_paradigms', t10)

# ================================================================
# Diagram 11: Generative vs Discriminative
# ================================================================
t11 = r'''
  \node[boxgreen,minimum width=3.5cm,minimum height=1.2cm] (gen) at (-2,1.5) {\textbf{Generative}\\\ $P(x,y)$ Joint Distribution};
  \node[lbl] at (-2,-1) {e.g. Naive Bayes, GANs\\ Good with limited labels\\ Can sample from distribution};
  \node[boxblue,minimum width=3.5cm,minimum height=1.2cm] (dis) at (3,1.5) {\textbf{Discriminative}\\\ $P(y|x)$ Conditional};
  \node[lbl] at (3,-1) {e.g. Logistic Reg, SVM, NN\\ More accurate with\\ sufficient labeled data};
'''
make_tex('bob_gen_disc', t11)

# ================================================================
# Diagram 12: Linear vs Logistic Regression
# ================================================================
t12 = r'''
  \node[boxgreen,minimum width=3cm] (lin) at (-2.5,2.5) {\textbf{Linear Regression}};
  \node[lbl] at (-2.5,0.5) {$y = w\cdot x + b$\\ Continuous output\\ MSE loss\\ Closed-form};
  \draw[fgcolor!60,line width=0.5pt] (-4.5,-1.2) -- (-0.5,-0.5);
  \foreach \x/\y in {-3.5/-1,-2.5/-0.7,-1.5/-0.3,-1/-0.6, -0.8/-0.4} {
    \fill[fgcolor!70] (\x,\y) circle (1pt);
  }
  \node[boxblue,minimum width=3cm] (log) at (3,2.5) {\textbf{Logistic Regression}};
  \node[lbl] at (3,0.5) {$p = \sigma(w\cdot x + b)$\\ Probability in $(0,1)$\\ Cross-entropy loss\\ Convex};
  \draw[fgcolor!60,line width=0.5pt] plot[domain=-0.5:1.5,samples=20] (\x*2.5+1, {1/(1+exp(-(\x)*4))-1.5});
  \node[lbl] at (-2.5,-1.8) {Line through points};
  \node[lbl] at (3,-1.8) {Sigmoid squashes to [0,1]};
'''
make_tex('bob_linear_logistic', t12)

# ================================================================
# Diagram 13: k-means clustering
# ================================================================
t13 = r'''
  \node[boxblue,minimum width=3cm] at (0,2.5) {\textbf{k-means Clustering}};
  \foreach \y/\l in {1.5/1. Pick $k$ centroids,0.3/2. Assign points,-0.9/3. Recomputed centroids,-2.1/4. Repeat 2-3} {
    \node[boxblue,minimum width=3cm,minimum height=0.5cm,font=\tiny] at (0,\y) {$\l$};
    \draw[arr] (0,\y-0.25) -- (0,\y-0.55);
  }
  \node[boxyellow,minimum width=3cm,font=\tiny] at (0,-3) {Choose $k$: Elbow / Silhouette};
  \foreach \cx/\cy/\c in {0.3/1.4/blue,-0.5/0.2/red,0.8/0.5/green} {
    \fill[\c!60] (\cx,\cy) circle (2pt);
    \fill[\c!60] (\cx+0.1,\cy+0.1) circle (2pt);
    \fill[\c!60] (\cx-0.1,\cy-0.1) circle (2pt);
  }
'''
make_tex('bob_kmeans', t13)

# ================================================================
# Diagram 14: Backpropagation
# ================================================================
t14 = r'''
  \node[boxgreen,minimum width=0.8cm] (x) at (-3,0) {$x$};
  \foreach \x/\l in {-1/Layer 1,1/Layer 2,3/Loss} {
    \node[boxblue,minimum width=1.2cm] at (\x,0) {$\l$};
    \draw[arr] (\x-1.3,0) -- (\x-0.6,0);
  }
  \foreach \x in {-1,1,3} {
    \draw[darr,fgcolor!50] (\x,-0.5) -- (\x-2,-0.5);
  }
  \node[lbl] at (0,-1) {Forward (solid) $\rightarrow$  Backward (dashed) $\leftarrow$};
  \node[lbl] at (0,-1.8) {Chain rule: $\partial L/\partial w = \partial L/\partial y \cdot \partial y/\partial z \cdot \partial z/\partial w$};
'''
make_tex('bob_backprop', t14)

# ================================================================
# Diagram 15: Feature Store
# ================================================================
t15 = r'''
  \node[boxgreen,minimum width=2.5cm] (train) at (-1.5,2) {Training Pipeline};
  \node[boxorange,minimum width=2.5cm] (serve) at (2,2) {Serving Pipeline};
  \node[boxyellow,minimum width=5cm,minimum height=1.8cm,line width=1.2pt] (fs) at (0,-1) {\textbf{FEATURE STORE}\\\ Feast / Tecton\\\ \\[-2pt] \footnotesize Offline: batch\\ \footnotesize Online: low-latency KV\\ \footnotesize Point-in-time correctness};
  \draw[arr] (train) -- (fs);
  \draw[arr] (serve) -- (fs);
  \node[lbl,fgcolor!70!green] at (0,-3) {No training/serving skew!};
'''
make_tex('bob_feature_store', t15)

# ================================================================
# Diagram 16: ML Project Lifecycle
# ================================================================
t16 = r'''
  \node[title] at (0,3.5) {ML Project Lifecycle};
  \foreach \x/\y/\l/\c in {
    -0.5/2.2/Frame\\Problem/boxblue,
    2.5/0.8/Data\\Collect/boxgreen,
    2.5/-1.2/Model\\Build/boxyellow,
    0/-2.8/Evaluate/boxorange,
    -2.5/-1.2/Deploy/boxpurple,
    -2.5/0.8/Monitor/boxpink
  } {
    \node[\c,minimum width=1.8cm,minimum height=1cm,font=\tiny] at (\x,\y) {$\l$};
  }
  \draw[arr] (-0.5,1.5) -- (2.5,1.3);
  \draw[arr] (2.5,0.2) -- (2.5,-0.2);
  \draw[arr] (2.5,-1.8) -- (0,-2.3);
  \draw[arr] (0,-3.3) -- (-2.5,-1.8);
  \draw[arr] (-2.5,-0.5) -- (-2.5,0);
  \draw[arr] (-2.5,1.5) -- (-0.5,1.8);
'''
make_tex('bob_ml_lifecycle', t16)

# ================================================================
# Diagram 17: STAR Framework
# ================================================================
t17 = r'''
  \node[title] at (0,3.5) {STAR Story Framework};
  \foreach \y/\l/\c in {2.5/SITUATION\\Context/boxblue,1/TASK\\Responsibility/boxgreen,-0.5/ACTION\\What done? (quantify)/boxyellow,-2/RESULT\\Outcome (metrics)/boxorange} {
    \node[\c,minimum width=3cm,minimum height=1cm] at (0,\y) {$\l$};
  }
  \draw[arr] (0,2) -- (0,1.5);
  \draw[arr] (0,0.5) -- (0,0);
  \draw[arr] (0,-1) -- (0,-1.5);
  \node[lbl] at (0,-3) {Prepare 3-4 stories for behavioral rounds};
'''
make_tex('bob_star', t17)

# ================================================================
# Diagram 18: Framing ML Problems
# ================================================================
t18 = r'''
  \node[boxorange,minimum width=3.5cm] (bp) at (0,3) {\small\textbf{Business Problem}\\\ \footnotesize "increase revenue"};
  \foreach \x/\l in {-2.5/Objective\\Metric?,-1/Scale\\Req/sec?,1/Latency\\Budget?,2.5/Data\\Available?} {
    \node[boxblue,minimum width=1.5cm,minimum height=0.8cm,font=\tiny] (q\x) at (\x,1.5) {$\l$};
    \draw[arr] (bp) -- (q\x);
    \draw[arr] (q\x) -- (0,0.5);
  }
  \node[boxgreen,minimum width=3cm,minimum height=0.8cm] (ml) at (0,-0.2) {\footnotesize Predict purchase probability};
  \node[boxyellow,minimum width=2cm] (tgt) at (-1.2,-1.5) {Target\\ + Metric};
  \node[boxyellow,minimum width=2cm] (bsl) at (1.2,-1.5) {Baseline\\ + Success};
  \draw[arr] (ml) -- (tgt); \draw[arr] (ml) -- (bsl);
'''
make_tex('bob_framing_ml', t18)

# ================================================================
# Diagram 19: Batch vs Online Inference
# ================================================================
t19 = r'''
  \node[title] at (-2,2) {Batch Inference};
  \node[boxblue] (dl) at (-2,0.5) {Data Lake};
  \node[boxgreen] (pre) at (-2,-1) {Precompute\\ Predictions};
  \draw[arr] (dl) -- (pre) node[midway,right,lbl] {Nightly};
  \node[boxyellow] (kv) at (-2,-2.5) {Key-Value\\ Store};
  \draw[arr] (pre) -- (kv);
  \node[boxorange] (sv) at (-2,-3.8) {Serve};

  \node[title] at (3,2) {Online Inference};
  \node[boxblue] (req) at (3,0.5) {Request};
  \node[boxorange] (md) at (3,-1) {Model\\ $<$100ms};
  \draw[arr] (req) -- (md) node[midway,right,lbl] {Real-time};
  \node[boxgreen] (sv2) at (3,-2.5) {Serve};
  \draw[arr] (md) -- (sv2);
'''
make_tex('bob_batch_online', t19)

# ================================================================
# Diagram 20: CI/CD Pipeline
# ================================================================
t20 = r'''
  \foreach \x/\l/\c in {0/Git\\Push/boxgreen,1.4/Data\\Valid/boxblue,2.8/Train\\Tune/boxyellow,4.2/Eval\\Offline/boxorange,5.6/Canary\\1\%/boxpurple,7/Shadow\\50\%/boxpink,8.4/Full\\Rollout/boxgreen} {
    \node[\c,minimum width=1cm,minimum height=0.8cm,font=\tiny] at (\x,0) {$\l$};
  }
  \foreach \x in {0.7,2.1,3.5,4.9,6.3,7.7} {
    \draw[arr] (\x,0) -- (\x+0.7,0);
  }
  \draw[darr,fgcolor!50] (4.2,-0.5) -- (0,-0.5) node[midway,below,lbl] {Fail $\rightarrow$ re-iterate};
  \draw[darr,fgcolor!50] (8.4,-0.5) -- (7,-0.5) node[midway,below,lbl] {Degraded $\rightarrow$ rollback};
'''
make_tex('bob_cicd', t20)

# ================================================================
# Diagram 21: Monitoring Pillars
# ================================================================
t21 = r'''
  \node[title] at (0,3) {Monitoring Pillars};
  \foreach \x/\l/\c in {-3/Model\\Performance\\Acc, AUC/boxblue,0/Data\\Quality\\PSI, Outliers/boxgreen,3/System\\Health\\Latency/boxyellow} {
    \node[\c,minimum width=2cm,minimum height=1.2cm,font=\tiny] at (\x,1) {$\l$};
  }
  \node[boxorange,minimum width=5cm,minimum height=0.8cm] (db) at (0,-1) {Dashboard: Prometheus + Grafana};
  \node[boxpink,minimum width=3cm] (al) at (0,-2.5) {Alert + Auto-Rollback};
  \draw[arr] (db) -- (al);
'''
make_tex('bob_monitoring', t21)

# ================================================================
# Diagram 22: Parallelism
# ================================================================
t22 = r'''
  \node[title] at (-2.5,3) {Data Parallelism};
  \node[boxblue] (b) at (-2.5,1.5) {Batch};
  \foreach \x/\l in {-3.8/Shard 1,-2.5/Shard 2,-1.2/Shard 3} {
    \node[boxgreen,minimum width=1.2cm,font=\tiny] at (\x,0.3) {$\l$};
    \draw[arr] (b) -- (\x,0.3);
    \node[boxyellow,minimum width=1.2cm,font=\tiny] at (\x,-1.2) {Model\\ Replica};
    \draw[arr] (\x,0.3) -- (\x,-1.2);
  }
  \node[boxorange,minimum width=2.5cm] (sg) at (-2.5,-2.8) {Sync Gradients};

  \node[title] at (3,3) {Model / Pipeline Parallelism};
  \foreach \x/\l in {2/Layers 1-3\\GPU 1,3.5/Layers 4-6\\GPU 2,5/Layers 7-9\\GPU 3} {
    \node[boxyellow,minimum width=1.2cm,font=\tiny] at (\x,0.5) {$\l$};
    \draw[arr] (\x+0.6,0.5) -- (\x+1.1,0.5);
  }
  \node[lbl] at (3.5,-1.5) {Split model stages across devices};
'''
make_tex('bob_parallelism', t22)

# ================================================================
# Diagram 23: Recommendation System
# ================================================================
t23 = r'''
  \node[boxblue] (user) at (-1.5,2.5) {User};
  \node[boxblue] (item) at (2,2.5) {Item};
  \node[boxgreen] (ut) at (-1.5,1) {User Tower};
  \node[boxgreen] (it) at (2,1) {Item Tower};
  \draw[arr] (user) -- (ut); \draw[arr] (item) -- (it);
  \node[boxyellow] (dot) at (0,-0.5) {Dot Product\\ Similarity};
  \draw[arr] (ut) -- (dot); \draw[arr] (it) -- (dot);
  \node[boxorange] (ann) at (0,-2) {ANN Retrieval\\ Top-N};
  \draw[arr] (dot) -- (ann);
  \node[boxgreen] (sv) at (0,-3.2) {Serve};
  \draw[arr] (ann) -- (sv);
'''
make_tex('bob_recommendation', t23)

# ================================================================
# Compile all
# ================================================================
tex_files = [f for f in os.listdir(OUT) if f.startswith('bob_') and f.endswith('.tex')]
tex_files.sort()
for f in tex_files:
    compile(os.path.join(OUT, f))

print(f"\nDone. Generated {len(tex_files)} TikZ SVGs in {OUT}")
