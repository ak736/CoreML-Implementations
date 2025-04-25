import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Part 2: Implementation of AdaBoost with decision trees as weak learners


class AdaBoost:
    def __init__(self, n_estimators=60, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.betas = []
        self.models = []

    def fit(self, X, y):
        ########################### TODO#############################################
        # In this part, please implement the adaboost fitting process based on the
        # lecture and update self.betas and self.models, using decision trees with
        # the given max_depth as weak learners

        # Inputs: X, y are the training examples and corresponding (binary) labels

        # Hint 1: remember to convert labels from {0,1} to {-1,1}
        # Hint 2: DecisionTreeClassifier supports fitting with a weighted training set
        n_samples = X.shape[0]

        # Initialize weights uniformly
        weights = np.ones(n_samples) / n_samples

        # Convert labels from {0,1} to {-1,1}
        y_transformed = np.where(y == 0, -1, 1)

        for _ in range(self.n_estimators):
            # Train a weak learner with weighted samples
            model = DecisionTreeClassifier(max_depth=self.max_depth)
            model.fit(X, y_transformed, sample_weight=weights)

            # Make predictions
            predictions = model.predict(X)

            # Calculate weighted error
            incorrect = predictions != y_transformed
            error = np.sum(weights * incorrect) / np.sum(weights)

            # Stop if error is 0 or >= 0.5
            if error <= 0:
                # For a perfect classifier, we'd use a very large beta
                # Since beta = 0.5 * ln((1-0)/0) approaches infinity
                self.models.append(model)
                # Use a large value for beta
                self.betas.append(10)  # Large value to give strong weight
                break

            if error >= 0.5:
                # Skip this weak learner as it's not better than random
                continue

            # Calculate beta value correctly according to the AdaBoost formula
            # This is the key change - using the correct formula from the lecture
            beta = 0.5 * np.log((1 - error) / error)

            # Store the model and its beta value
            self.models.append(model)
            self.betas.append(beta)

            # Update weights using the formula from the lecture
            weights = weights * np.exp(-beta * y_transformed * predictions)

            # Normalize weights
            weights = weights / np.sum(weights)

        return self

    def predict(self, X):
        ########################### TODO#############################################
        # In this part, make prediction on X using the learned ensemble
        # Note that the prediction needs to be binary, that is, 0 or 1.

        if len(self.models) == 0:
            raise ValueError("Model has not been trained yet.")

        # Initialize predictions
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        # Calculate weighted sum of weak learners' predictions
        for model, beta in zip(self.models, self.betas):
            # Get predictions from the model
            predictions = model.predict(X)

            # Add the weighted predictions using beta directly
            scores += beta * predictions

        # Final prediction: sign of the weighted sum
        # Convert back from {-1,1} to {0,1}
        preds = np.where(scores < 0, 0, 1)

        return preds

    def score(self, X, y):
        accuracy = accuracy_score(y, self.predict(X))
        return accuracy
