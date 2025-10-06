
# Ablation CAM: Class Activation Mapping using Ablation

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A PyTorch implementation of Ablation Class Activation Mapping (Ablation-CAM), a visualization technique for understanding and interpreting deep neural network decisions.

## 📋 Table of Contents
- [Overview](#overview)
- [How Ablation-CAM Works](#how-ablation-cam-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Results](#results)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

Ablation-CAM is a gradient-free method for generating Class Activation Maps (CAM) that helps visualize which regions of an input image are most important for a model's prediction. Unlike gradient-based methods like Grad-CAM, Ablation-CAM uses a systematic ablation approach to measure the importance of each activation unit.

**Key Features:**
- ✅ No gradient computation required
- ✅ Works with any CNN architecture
- ✅ Provides faithful visual explanations
- ✅ Easy to implement and use

## 🛠 How Ablation-CAM Works

The Ablation-CAM algorithm:

1. **Forward Pass**: Pass the input image through the CNN to get class predictions
2. **Activation Extraction**: Extract feature maps from the target convolutional layer
3. **Systematic Ablation**: Iteratively ablate (set to zero) each channel in the feature maps
4. **Importance Scoring**: Measure the drop in target class score when each channel is ablated
5. **Weight Calculation**: Compute importance weights based on score reductions
6. **Map Generation**: Generate the final CAM by linearly combining feature maps with importance weights

Mathematically:
\[
\text{Ablation-CAM}(x,y) = \text{ReLU}\left(\sum_{k} w_k^y \cdot A^k(x)\right)
\]
where \(w_k^y = \frac{Y_c - Y_c^k}{Y_c}\) represents the importance of channel \(k\) for class \(y\).

## 💻 Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.7+
- torchvision
- OpenCV
- NumPy
- Matplotlib

### Install via pip
```bash
pip install torch torchvision opencv-python matplotlib numpy
