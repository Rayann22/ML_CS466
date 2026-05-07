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

## Project Structure

```text
ML_CS466/
│
├── code/
│   ├── dataset.py                    # Data loading and preprocessing
│   ├── model.py                      # Kim CNN model implementation
│   ├── train.py                      # Training and evaluation pipeline
│
├── datasets/
│   │
│   ├── IMDb_clean/
│   │   ├── rt-polarity.pos           # Positive IMDb reviews
│   │   └── rt-polarity.neg           # Negative IMDb reviews
│   │
│   ├── IMDb_grammar_modified/
│   │   ├── imdb_grammar.pos          # Grammar-modified positive reviews
│   │   └── imdb_grammar.neg          # Grammar-modified negative reviews
│   │
│   └── twitter_noisy/
│       ├── twitter_clean.pos         # Positive Twitter samples
│       └── twitter_clean.neg         # Negative Twitter samples
│
├── embeddings/
│   └── GoogleNews-vectors-negative300.bin.gz
│                                       # Pretrained Word2Vec embeddings
│
├── results/
│   │
│   ├── IMDb_clean_results/              # Results for clean IMDb dataset
│   ├── IMDb_grammar_modified_results/   # Results for grammar-modified IMDb                           
│   └── twitter_results/                 # Results for Twitter dataset
│
└── README.md
```

---

## Datasets

This project uses multiple sentiment classification datasets for evaluating CNN-based sentence classification models under different text conditions.

### IMDb Dataset (Clean Text)

The IMDb movie reviews dataset is used for clean sentiment classification experiments.

Download the dataset from:

https://ai.stanford.edu/~amaas/data/sentiment/

After downloading and extracting the dataset, place the files inside:

```text
datasets/imdb/
```

---

### Sentiment140 Twitter Dataset (Noisy Text)

The Sentiment140 dataset is used for noisy Twitter sentiment classification experiments.

Download the dataset from:

https://www.kaggle.com/datasets/kazanova/sentiment140

Place the dataset files inside:

```text
datasets/twitter/
```

---

### Grammar-Modified IMDb Dataset

The grammar-modified IMDb dataset is based on modified versions of IMDb reviews containing grammatical inconsistencies and altered sentence structures.

Place the processed dataset files inside:

```text
datasets/imdb_grammar/
```

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
mkdir embeddings
wget embeddings https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
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

1. Install the required dependencies:

```bash
pip install torch numpy scikit-learn gensim tqdm
```

2. Select the CNN model type in:

```text
code/train.py
```

Change the model type to one of the following options:

```text
rand
static
non-static
```

### Model Variants

- `rand`  
  CNN-rand model with randomly initialized embeddings.

- `static`  
  CNN-static model using pretrained Word2Vec embeddings that remain fixed during training.

- `non-static`  
  CNN-non-static model using pretrained Word2Vec embeddings that are fine-tuned during training.

---

3. Select the dataset in:

```text
code/train.py
```

Change the dataset path or dataset option to one of the following:

```text
datasets/imdb/
datasets/twitter/
datasets/imdb_grammar/
```

### Dataset Options

- `imdb`  
  Clean IMDb movie reviews dataset.

- `twitter`  
  Noisy Twitter sentiment dataset.

- `imdb_grammar`  
  Grammar-modified IMDb dataset.

---

4. Run the training script:

```bash
python code/train.py
```

---

## Reference

Yoon Kim. (2014).
*Convolutional Neural Networks for Sentence Classification*

---


