# Toxic Comment Classification using Context-Aware BERT Regression

This project implements a context-aware toxic comment detection system using transformer-based deep learning. Instead of binary toxic/non-toxic classification, the model predicts a continuous toxicity score (0–1) to better capture nuanced toxicity, sarcasm, and identity-related context.

The project fine-tunes a pretrained transformer model for toxicity regression and evaluates performance using ROC-AUC, precision–recall curves, and confusion matrices. It also includes a simple interactive UI for real-time comment evaluation.


## Problem Statement

Online platforms generate millions of user comments daily, and many contain toxic, abusive, or biased language. Manual moderation is not scalable. Traditional keyword-based or binary classifiers often:

Over-flag identity-related terms (e.g., “gay”, “Muslim”)
Fail to detect sarcasm and implicit toxicity
Lose nuance due to hard binary labels

This project addresses these issues by predicting toxicity on a continuous scale using a transformer-based model.

# Dataset

    Source: Jigsaw toxicity dataset (cleaned subset)
Key columns:

    Comment_text: input comment
    Toxicity: continuous toxicity score (0–1)

Preprocessing steps in the notebook:

    1. Remove missing values
    2. Normalize column names
    3. Split data into train/validation sets
## Model Architecture

The project experiments with transformer-based models including:

    - BERT-base-uncased as baseline
    - Improved model using transformer backbone with:
        Regression head (single output neuron)
        Continuous output score
        BCE / regression-style loss

Key implementation details from the notebook:

    - Tokenization with Hugging Face tokenizer
    - PyTorch Dataset & DataLoader pipeline
    - Adam/AdamW optimizer
    - Learning rate scheduling
## Training Pipeline

Main training steps:

    1. Load dataset and tokenize text
    2. Convert text to input IDs and attention masks
    3. Train model with:
        Batched DataLoader
        GPU acceleration (if available)
        Learning rate scheduler
    4. Save best model weights
## Training Pipeline

Main training steps:

    1. Load dataset and tokenize text
    2. Convert text to input IDs and attention masks
    3. Train model with:
        Batched DataLoader
        GPU acceleration (if available)
        Learning rate scheduler
    4. Save best model weights
## Evaluation

The model is evaluated using:

    ROC-AUC curve
    Precision–Recall curve
    Confusion matrix
    F1-score (threshold-based)

The notebook also includes visualization code for:

    ROC curve
    Precision–recall curve
    Confusion matrix heatmap
## Inference & Interactive Demo

The notebook includes an interactive UI built with ipywidgets that allows:

1. Entering custom text
2. Getting a toxicity score prediction
3. Interpreting output based on score ranges
## Technologies Used

    Python
    PyTorch
    Hugging Face Transformers
    Scikit-learn
    Matplotlib / Seaborn
    ipywidgets (for UI)
## How to run?


Open the notebook PRML_project.ipynb using Jupyter Notebook or Google Colab.

Install required libraries (if not already installed):

    pip install torch transformers scikit-learn pandas numpy matplotlib ipywidgets

Run all cells in order. The notebook will:
- Load and preprocess data
- Train the BERT-based regression model
- Evaluate performance
- Allow interactive testing of comments
## Project Structure

    PRML_project.ipynb #Main notebook containing code, training, and evaluation

    README.md  #Project documentation
## Authors

- [Niki](https://www.github.com/Niki-ux)
- [S Viveka](https://www.github.com/Viveka-S-2)
- [Annu Sevada](https://www.github.com/annusevada)
- [Dhatrik Jaikritha](https://www.github.com/jai-time29)


