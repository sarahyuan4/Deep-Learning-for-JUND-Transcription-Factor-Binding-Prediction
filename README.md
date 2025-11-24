# Deep Learning for JUND Transcription Factor Binding Prediction

This project applies supervised deep learning models to predict binding sites of the **JUND transcription factor (TF)** on human chromosome 22. The models use DNA sequence data, chromatin accessibility, and weighted labels to address class imbalance. Three neural architectures are implemented and compared: **MLP**, **CNN**, and **LSTM**.

---

## Overview

Transcription factors regulate gene expression by binding to specific DNA motifs. Accurately predicting TF binding reduces experimental cost and helps in understanding gene regulation.  
This work focuses on **JUND**, a TF associated with the AP-1 protein family.

Dataset includes:

- `X`: One-hot encoded DNA sequence, 101 bp (101 × 4)  
- `y`: Label (1 if JUND binds, 0 otherwise)  
- `w`: Weight per sample (used to correct class imbalance)  
- `a`: Chromatin accessibility value  

Only **0.42%** of samples are positive, therefore training uses **weighted binary cross-entropy loss**.

More information about JUND:  
https://www.genecards.org/cgi-bin/carddisp.pl?gene=JUND

---

## Models

The following neural architectures were implemented using PyTorch:

### 1. MLP (Multi-Layer Perceptron)
- Input: Flattened 101 × 4 sequence → 404-dimensional vector  
- Accessibility value `a` concatenated before the final layer  
- Serves as a strong, simple baseline

### 2. CNN (1D Convolutional Neural Network)
- Uses `Conv1d` over the sequence (treated as length-101 with 4 channels)  
- Learns motif-like filters that capture local sequence patterns  
- Output features flattened and passed to an MLP  
- Accessibility value `a` concatenated before the final prediction layer

### 3. LSTM (Recurrent Neural Network)
- Uses an LSTM encoder with `batch_first=True` on inputs of shape (N, 101, 4)  
- Last hidden state used as sequence embedding  
- Embedding fed into a two-layer MLP  
- Accessibility value `a` concatenated before the output layer  

All models:
- Use `BCEWithLogitsLoss` with sample weights (`w`)  
- Are trained with the Adam optimizer  
- Are evaluated on held-out validation and test sets

---

## Results (Weighted Accuracy on Test Set)

| Model | Weighted Accuracy |
|-------|-------------------|
| MLP   | ~0.64             |
| CNN   | ~0.64 (limited epochs) |
| LSTM  | ~0.51 (undertrained)   |

### Notes

- The **MLP baseline performs strongly**, even though it does not explicitly model sequence order.
- The **CNN** is expected to surpass the MLP with additional tuning (more epochs, filter sizes, number of channels, regularization).
- The **LSTM** performed worst under the current setup, likely due to insufficient training time; recurrent models generally require more epochs and careful hyperparameter tuning.
