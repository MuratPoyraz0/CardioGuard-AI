# 🧬 CardioGuard-AI: GNN-Based hERG Toxicity Predictor & XAI Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cardioguard-ai-app.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-blue)](https://www.rdkit.org/)

**CardioGuard-AI** is a deep learning-powered cheminformatics system designed to predict **hERG (human Ether-à-go-go-Related Gene) potassium channel blockade toxicity** in early-stage drug discovery. By converting molecular SMILES into graph neural network representations, CardioGuard-AI delivers both accurate toxicity classification and **Explainable AI (XAI)** atom-level attribution heatmaps.

---

## 🎯 Key Features

- **Graph Neural Network Architecture:** Built on a customized **Graph Isomorphism Network (GIN)** optimized for graph-level molecular property prediction.
- **Explainable AI (XAI):** Generates **Similarity Attribution Heatmaps** using RDKit drawing engines to highlight specific atomic structures contributing to cardiotoxicity.
- **Cheminformatics Pipeline:** Automates standard molecular parsing (SMILES) to graph conversion with 5-dimensional atomic node features (Atomic Number, Degree, Formal Charge, Aromaticity, Total H Count).
- **Interactive Web Interface:** Fully deployed via Streamlit Cloud for real-time inference and dynamic visualization.

---

## 📊 Benchmark Metrics (TDC hERG Dataset)

Evaluated on the benchmark **Therapeutics Data Commons (TDC)** hERG Target Dataset:

| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | **84.2%** |
| **Accuracy** | **81.5%** |
| **Precision** | **79.8%** |
| **F1-Score** | **80.6%** |

---

## 🛠️ System Architecture & Tech Stack

- **Core Framework:** Python 3.11, PyTorch 2.x
- **Graph Processing:** PyTorch Geometric (PyG)
- **Cheminformatics:** RDKit (`Chem`, `Draw`, `SimilarityMaps`)
- **Deployment & UI:** Streamlit Cloud

### Graph Pipeline Overview
```text
SMILES String ➔ RDKit Molecule Object ➔ Graph Extraction (Nodes: 5 Atom Features, Edges: Bonds) ➔ GIN Layers + Global Mean Pooling ➔ Softmax Risk Probability
