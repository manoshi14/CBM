# Concept Alignment for Interpretable Bangla Text

### A Concept Bottleneck Model Approach to Bangla Sarcasm Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## Overview

This repository contains the implementation of the first Concept Bottleneck Model (CBM) framework for Bangla Natural Language Processing.

The project introduces an interpretable sarcasm detection pipeline for Bangla that combines transformer language models, unsupervised concept discovery, large language model-assisted concept labeling, and human-annotated concept supervision.

Unlike conventional black-box classifiers, every prediction produced by the model is routed through a set of human-interpretable concepts, enabling transparent and auditable decision making.

## Main Contributions

* First application of Concept Bottleneck Models (CBMs) to Bangla NLP.
* Data-driven discovery of Bangla sarcasm concepts using UMAP and HDBSCAN clustering.
* Construction of a human-curated concept bank containing 43 interpretable concepts.
* Creation of the first large-scale human-annotated concept supervision matrix for Bangla sarcasm detection.
* Benchmarking of 9 transformer backbones across 3 learning paradigms (27 total experiments).
* Empirical demonstration that human concept supervision substantially outperforms automated cosine-similarity supervision.

---

## Dataset

We use the BenSarc dataset for Bangla sarcasm detection.

| Statistic                 | Value   |
| ------------------------- | ------- |
| Original Samples          | 12,819  |
| Filtered Samples          | 10,164  |
| Training Samples          | 8,131   |
| Test Samples              | 2,033   |
| Concepts                  | 43      |
| Human Concept Annotations | 437,052 |

The dataset covers multiple domains including:

* Politics
* Sports
* Religion
* Entertainment
* Lifestyle
* Social Commentary

---

## Methodology

The proposed framework consists of seven stages:

1. Label-specific text clustering
2. BanglaBERT embedding extraction
3. UMAP dimensionality reduction
4. HDBSCAN cluster discovery
5. LLM-assisted concept labeling
6. Concept bank consolidation
7. Concept Bottleneck Model training

### Concept Discovery Pipeline

```text
Bangla Text
      ↓
BanglaBERT Embeddings
      ↓
UMAP
      ↓
HDBSCAN
      ↓
Cluster Groups
      ↓
LLM Concept Labels
      ↓
Concept Consolidation
      ↓
43 Final Concepts
      ↓
Concept Bottleneck Model
      ↓
Sarcasm Prediction
```

---

## Models Evaluated

### Bangla-Specific Models

* BanglaBERT
* Bangla Sentence Transformer

### Indic Models

* MuRIL
* IndicBERTv2 MLM
* IndicBERTv2 MLM + Sampled TLM

### Multilingual Models

* mBERT
* XLM-RoBERTa
* Multilingual MPNet
* mT5

---

## Experimental Paradigms

### 1. Direct Label Classification

Standard transformer fine-tuning on sarcasm labels.

### 2. CBM + Cosine Similarity Supervision

Concept activations generated automatically using embedding similarity.

### 3. CBM + Manual Concept Supervision

Concept activations learned from human annotations.

---

## Key Results

| Paradigm                 | Best Model                    | Macro-F1 |
| ------------------------ | ----------------------------- | -------- |
| Direct Classification    | IndicBERTv2 MLM               | 0.7689   |
| CBM + Cosine Supervision | XLM-RoBERTa                   | 0.6513   |
| CBM + Manual Supervision | IndicBERTv2 MLM + Sampled TLM | 0.7600   |

### Main Finding

Human-annotated concept supervision consistently outperforms automated cosine-similarity concept scoring across all evaluated transformer backbones.

This result suggests that embedding similarity alone is insufficient for capturing the pragmatic and cultural nuances of Bangla sarcasm.

---

## Repository Structure

```text
.
├── data/
├── concept_discovery/
├── clustering/
├── annotations/
├── cbm_models/
├── baseline_models/
├── experiments/
├── results/
├── figures/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/bangla-cbm-sarcasm.git

cd bangla-cbm-sarcasm

pip install -r requirements.txt
```

---

## Running Experiments

### Direct Classification

```bash
python train_direct_classifier.py
```

### CBM with Manual Concept Supervision

```bash
python train_cbm_manual.py
```

### CBM with Cosine Similarity Supervision

```bash
python train_cbm_cosine.py
```

---

## Reproducibility

All experiments were conducted using:

* Python 3.10+
* PyTorch 2.x
* HuggingFace Transformers
* UMAP
* HDBSCAN
* Scikit-Learn

Random seed:

```python
seed = 42
```

---

## Citation

If you use this repository, please cite:

```bibtex
@thesis{concept_alignment_bangla_2026,
  title={Concept Alignment for Interpretable Bangla Text:
         A Concept Bottleneck Model Approach to Sarcasm Detection},
  author={Your Name},
  year={2026}
}
```

---

## Acknowledgements

This work builds upon:

* BenSarc Dataset
* BanglaBERT
* MuRIL
* IndicBERTv2
* XLM-RoBERTa
* Concept Bottleneck Models (Koh et al., 2020)
* CB-LLMs Framework

---

## License

MIT License

```

For research visibility, I'd also rename the repository to one of:

- `Bangla-CBM-Sarcasm`
- `ConceptAlignment-Bangla`
- `Interpretable-Bangla-NLP`
- `Bangla-Concept-Bottleneck-Models`

The strongest academic-looking name is **Bangla-CBM-Sarcasm**.  
```
