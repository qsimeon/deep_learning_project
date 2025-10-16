# Multimodal Alignment: Towards the Platonic Representation

This project implements a framework for aligning pre-trained unimodal encoders (ResNet-18 for images, DistilBERT for text) into a shared multimodal latent space using lightweight adapters. The goal is to test whether such aligned representations better approximate those of larger, more performant models like DINOv2, providing empirical evidence for the Platonic Representation Hypothesis.

## Quick Start

1. **Setup Environment**:
   ```bash
   conda create -n multimodal-align python=3.11
   conda activate multimodal-align
   pip install -r requirements.txt
   ```

2. **Run the Experiment**:
   ```bash
   jupyter notebook multimodal_alignment.ipynb
   ```
   Then execute all cells to run the complete experiment.

## Key Features

- **Frozen Encoders**: ResNet-18 (ImageNet) and DistilBERT (BookCorpus) remain frozen during training
- **Lightweight Adapters**: Linear and MLP transformations to shared embedding space
- **Contrastive Learning**: Dual-encoder contrastive loss for multimodal alignment
- **Comprehensive Evaluation**: Kernel alignment metrics, similarity analysis, and downstream CIFAR-10 evaluation
- **Robust Data Loading**: Automatic fallbacks for dataset/model loading issues

## Experiment Results

The notebook will run experiments with both Linear and MLP adapters, comparing:

1. **Kernel Alignment**: How well aligned representations match DINOv2 embeddings
2. **Cross-Modal Similarity**: Image-text alignment quality
3. **Downstream Performance**: CIFAR-10 classification accuracy

## Hypothesis Testing

The framework tests three key hypotheses:

1. **Shared Latent Space**: Can unimodal representations be aligned through linear transformations?
2. **Platonic Representation**: Do aligned representations approximate those of performant models?
3. **Representation Convergence**: Does multimodal alignment capture convergence mechanisms?

## Files

- `multimodal_alignment.ipynb`: Main experiment notebook
- `requirements.txt`: Python dependencies
- `index.html`: Original blog post with results and analysis
- `figures/`: Visualization outputs and diagrams

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- ~2GB disk space for models and datasets

## Notes

- The notebook is designed to run on CPU or GPU
- Dataset loading includes automatic retries and fallbacks
- Results are automatically saved to `./results/` directory
- All models are loaded with `token=None` to avoid authentication issues
