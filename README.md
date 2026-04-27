# CNN for Sentence Classification

##  Overview

This project implements a Convolutional Neural Network (CNN) for sentence classification based on the paper:

**Yoon Kim (2014) – "Convolutional Neural Networks for Sentence Classification"**

The goal is to reproduce Kim’s model and evaluate its performance on a modified dataset derived from IMDb reviews, while preserving the original MR-style data format.

---

##  Objectives

* Reproduce the CNN architecture proposed by Kim (2014)
* Apply the model to IMDb-based data formatted like the MR dataset
* Compare different embedding modes:

  * Random (rand)
  * Static (pre-trained, fixed)
  * Non-static (pre-trained, trainable)
* Evaluate performance using multiple metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

##  Project Structure

```
ML_CS466/
│
├── code/
│   ├── dataset.py        # Data loading and preprocessing
│   ├── model.py          # Kim CNN model implementation
│   ├── train.py          # Training and evaluation pipeline
│
├── data/
│   ├── rt-polarity.pos   # Positive samples
│   ├── rt-polarity.neg   # Negative samples
│
├── results/
│   ├── rand_results.txt
│   ├── static_results.txt
│   ├── non-static_results.txt
│
└── README.md
```

---

##  Dataset

The dataset is derived from the IMDb dataset but converted into the same format used in Kim (2014):

* Two separate files:

  * `rt-polarity.pos` (positive reviews)
  * `rt-polarity.neg` (negative reviews)
* Each line represents one review
* A balanced subset of **1000 samples** is used:

  * 500 positive
  * 500 negative

This allows fair comparison with the original MR dataset structure.

---

## Model Architecture

The model follows Kim’s CNN design:

* Embedding Layer (random or pre-trained Word2Vec)
* Multiple convolution filters (sizes 3, 4, 5)
* Max-over-time pooling
* Dropout layer
* Fully connected layer
* Softmax output

---

## Training Setup

* Optimizer: Adadelta
* Loss Function: CrossEntropyLoss
* Batch size: 32
* Epochs: 6
* Evaluation: 10-fold cross-validation

---

## Evaluation Metrics

The model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**


---
## Pre-trained Embeddings (Required)

This project uses **pre-trained Word2Vec embeddings** from Google News for the *static* and *non-static* models.

###  Download

You must download the file:

```text
GoogleNews-vectors-negative300.bin.gz
```

From:
https://code.google.com/archive/p/word2vec/

or
```bash
cd embeddings
wget -P embeddings https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
```

###  Placement

After downloading, place the file in the following directory:

```text
embeddings/GoogleNews-vectors-negative300.bin.gz
```

### Important Notes

* File size ≈ **1.5 GB**
* Required only for:

  * `static`
  * `non_static`
* Not required for:

  * `rand` model

---

## How to Run

1. Make sure dependencies are installed:

```
pip install torch numpy scikit-learn gensim tqdm
```

2. Run training:

```
python code/train.py
```

---

## Reference

Yoon Kim. (2014).
*Convolutional Neural Networks for Sentence Classification*

---


