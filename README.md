# Neural Network Classification of ASD vs. TD Using Linguistic Features and Zipf’s Law

## Overview

This research project explores the use of Natural Language Processing (NLP) and neural networks to distinguish between Autism Spectrum Disorder (ASD) and Typically Developing (TD) individuals using linguistic patterns in text samples.

The model combines engineered linguistic features with statistical properties derived from Zipf’s Law to analyze differences in word frequency distributions and language structure.

This research was conducted under mentorship through the Lumiere Education AI Research Program.

---

## Research Motivation

Language patterns may differ between ASD and TD populations. Prior studies suggest differences in:

- Lexical diversity  
- Word frequency distributions  
- Structural and syntactic patterns  

Zipf’s Law provides a mathematical framework for modeling word frequency scaling in natural language. By integrating Zipfian analysis with traditional linguistic features, this project aims to improve classification performance.

---

## Methodology

### 1. Data Processing
- Cleaned and tokenized text samples  
- Standardized formatting  
- Removed noise and irrelevant tokens  

### 2. Feature Engineering
Extracted quantitative linguistic features including:
- Lexical diversity metrics  
- Average sentence length  
- Word frequency statistics  
- Zipf’s Law slope and goodness-of-fit measures  

### 3. Model Development
- Implemented a neural network classifier  
- Compared performance against baseline models (e.g., logistic regression)  
- Tuned hyperparameters for improved generalization  

### 4. Evaluation
- Train/test split  
- Accuracy, precision, recall, and F1-score metrics  
- Comparative performance analysis  

---

## Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib / Seaborn  

---

## Results

The neural network demonstrated improved classification performance compared to baseline statistical models, suggesting that combining linguistic feature engineering with Zipfian statistical modeling can enhance ASD vs. TD differentiation.

*Note: Results are research-based and not intended for clinical diagnosis.*

---

## Limitations

- Dataset size constraints  
- Potential sampling bias  
- Limited generalizability  

Future work may explore transformer-based architectures and larger datasets.

---

## Research Paper

This project resulted in a published research paper in the American Journal of Student Research (AJOSR) on the limits of linguistic and Zipfian features for autism classification using neural networks.

---

## Disclaimer

This project is for academic research purposes only and does not constitute medical or diagnostic advice.
