# Core ML Implementations

A comprehensive collection of machine learning algorithms implemented from scratch. This repository contains pure implementations of fundamental ML techniques without relying on high-level libraries. Each implementation includes detailed documentation and examples to help understand the underlying concepts.

Perfect for ML enthusiasts, students, and practitioners who want to deepen their understanding of how these algorithms work at their core.

## Algorithms Implemented

### Classification

- **KNN (K-Nearest Neighbors)**
  - Implementation of the k-nearest neighbors algorithm for classification tasks
  - Distance metrics: Euclidean, Manhattan, Minkowski
  - Weighted and unweighted voting options

- **Decision Trees**
  - Implementation of classification and regression trees
  - Splitting criteria: Gini impurity, Entropy, Information gain
  - Pruning techniques to avoid overfitting

- **Boosting**
  - AdaBoost implementation
  - Gradient Boosting implementation
  - Features importance calculation

- **Linear Classifier**
  - Logistic Regression with gradient descent optimization
  - Support Vector Machines with different kernel options
  - Perceptron algorithm implementation

### Clustering

- **K-Means**
  - Implementation of the k-means clustering algorithm
  - Different initialization methods (random, k-means++)
  - Silhouette analysis for optimal k selection

### Dimensionality Reduction

- **PCA (Principal Component Analysis)**
  - Implementation of PCA for dimensionality reduction
  - Visualization tools for principal components
  - Variance explanation ratio calculation

### Neural Networks

- **Neural Nets**
  - Feedforward Neural Network implementation
  - Backpropagation algorithm from scratch
  - Various activation functions (ReLU, Sigmoid, Tanh)
  - Different optimizers (SGD, Adam, RMSProp)

- **Transformers**
  - Self-attention mechanism implementation
  - Multi-head attention implementation
  - Positional encoding
  - Transformer encoder and decoder blocks

### Reinforcement Learning

- **Q-Learning**
  - Implementation of Q-learning algorithm
  - Experience replay mechanism
  - Exploration-exploitation strategies

### Sequential Models

- **HMM (Hidden Markov Models)**
  - Implementation of Hidden Markov Models
  - Forward-backward algorithm
  - Viterbi algorithm for most likely state sequence
  - Baum-Welch algorithm for parameter estimation

## Structure

Each algorithm implementation includes:

- Core implementation file with detailed comments
- Jupyter notebook with example usage and visualizations
- Test cases to verify correctness
- Documentation explaining the mathematical foundations

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Jupyter (for notebooks)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.