# Amazon Reviews Classification

## Overview
A deep learning project that classifies Amazon product reviews as positive or negative using LSTM neural networks. Built with TensorFlow/Keras, this binary sentiment classifier achieves high accuracy in determining customer sentiment from review text.

## Problem Statement
Automating sentiment analysis of customer reviews to help businesses quickly understand product reception and customer satisfaction at scale, eliminating manual review analysis.

## Dataset
- **Source**: Amazon Reviews Dataset (50,000 reviews)
- **Features**: Review text and sentiment labels (positive/negative)
- **Split**: 80% training, 20% testing
- **Balance**: Evenly distributed positive and negative samples

## Technical Approach

### 1. Text Preprocessing
- Tokenization using Keras Tokenizer (vocabulary size: 10,000 words)
- Sequence padding to uniform length (max_length: 200)
- Text cleaning and normalization

### 2. Model Architecture
- **Embedding Layer**: 128-dimensional word embeddings
- **LSTM Layer**: 64 units with dropout (0.5) for regularization
- **Dense Output**: Sigmoid activation for binary classification
- **Optimizer**: Adam with binary cross-entropy loss

### 3. Training Configuration
- Epochs: 5
- Batch Size: 64
- Validation Split: 20%
- Early stopping to prevent overfitting

## Results
- **Test Accuracy**: ~87-89%
- **Model Performance**: Strong generalization on unseen reviews
- **Convergence**: Stable training with minimal overfitting

## Technologies Used
- **Framework**: TensorFlow 2.x / Keras
- **Language**: Python 3.x
- **Libraries**: NumPy, Pandas, NLTK
- **Architecture**: Sequential LSTM

## Usage

### Prerequisites
```bash
pip install tensorflow pandas numpy nltk
```

### Running the Project
```bash
python amazon_reviews_classification.py
```

The script will:
1. Load and preprocess the Amazon reviews dataset
2. Build and compile the LSTM model
3. Train for 5 epochs with validation
4. Evaluate on test set and display accuracy
5. Save the trained model for deployment

## Key Learnings
- Implemented sequential LSTM architecture for NLP classification
- Applied dropout regularization to improve generalization
- Handled large-scale text data preprocessing pipeline
- Optimized hyperparameters for sentiment analysis task

## Future Enhancements
- Experiment with bidirectional LSTM/GRU architectures
- Implement attention mechanisms for better context understanding
- Add multi-class classification for rating prediction (1-5 stars)
- Deploy as REST API for real-time sentiment prediction

## Author
Vinamra Jha
