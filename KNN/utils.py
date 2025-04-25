import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)

    # Count  TP, FP, FN
    tp = 0  # True Positive
    fp = 0  # True Negative
    fn = 0  # False Negative

    for real, pred in zip(real_labels, predicted_labels):
        if pred == 1:  # Predicted Positive
            if real == 1:  # True Positive
                tp += 1
            else:  # False Positive
                fp += 1
        elif pred == 0 and real == 1:  # Predicted Negative but was Positive
            fn += 1  # False Negative

    # If precision  and recall are zero, then
    if tp == 0:
        return 0.0

    # Calculate Precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Calculate F1 score
    f1_score_value = 2 * (precision * recall) / (precision + recall)

    return f1_score_value


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        # Convert to numpy arrays
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Calculate the absolute difference between point1 and point2 to the power 3 and sum it
        absolute_diff = np.sum(np.abs(point1 - point2) ** 3)

        return absolute_diff ** (1/3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        # Convert to numpy arrays
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Calculate the squared difference between point1 and point2 and sum it
        squared_diff = np.sum(np.abs(point1 - point2) ** 2)

        return np.sqrt(squared_diff)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        # Convert to numpy arrays
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Check if points are zero vector
        if np.all(point1 == 0) or np.all(point2 == 0):
            return 1.0

        # Calculate the cosine similarity
        dot_product = np.dot(point1, point2)
        magnitude_product = np.linalg.norm(point1) * np.linalg.norm(point2)
        cosine_similarity = 1 - (dot_product / magnitude_product)

        # Return the cosine distance cosine similarity
        return cosine_similarity


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
                (this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        # Initialize the best score
        best_f1 = 0
        # Priority order dictionary
        distance_priority = {
            'euclidean': 0,
            'minkowski': 1,
            'cosine_dist': 2
        }

        # First find best F1 score
        for k in range(1, 30, 2):
            for dist_name, dist_func in distance_funcs.items():
                model = KNN(k=k, distance_function=dist_func)
                model.train(x_train, y_train)
                predictions = model.predict(x_val)
                current_f1 = f1_score(y_val, predictions)

                # Update if better score OR equal score with higher priority
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    self.best_k = k
                    self.best_distance_function = dist_name
                    self.best_model = model
                elif current_f1 == best_f1:
                    # If equal F1 score, apply priority rules
                    curr_priority = distance_priority[dist_name]
                    best_priority = distance_priority[self.best_distance_function] if self.best_distance_function else float(
                        'inf')

                    if curr_priority < best_priority or (curr_priority == best_priority and k < self.best_k):
                        self.best_k = k
                        self.best_distance_function = dist_name
                        self.best_model = model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """

        # Initialize the best score
        best_f1 = 0
        distance_priority = {
            'euclidean': 0,
            'minkowski': 1,
            'cosine_dist': 2
        }
        scaler_priority = {
            'min_max_scale': 0,
            'normalize': 1
        }

        # Try each scaler in priority order
        for scaler_name, ScalerClass in scaling_classes.items():
            scaler = ScalerClass()
            scaled_x_train = scaler(x_train)
            scaled_x_val = scaler(x_val)

            for k in range(1, 30, 2):
                for dist_name, dist_func in distance_funcs.items():
                    model = KNN(k=k, distance_function=dist_func)
                    model.train(scaled_x_train, y_train)
                    predictions = model.predict(scaled_x_val)
                    current_f1 = f1_score(y_val, predictions)

                    # Update if better score OR equal score with higher priority
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        self.best_k = k
                        self.best_distance_function = dist_name
                        self.best_scaler = scaler_name
                        self.best_model = model
                    elif current_f1 == best_f1:
                        # Check priorities in order: scaler -> distance -> k
                        curr_scaler_priority = scaler_priority[scaler_name]
                        best_scaler_priority = scaler_priority[self.best_scaler] if self.best_scaler else float(
                            'inf')

                        if (curr_scaler_priority < best_scaler_priority) or \
                           (curr_scaler_priority == best_scaler_priority and
                            distance_priority[dist_name] < distance_priority[self.best_distance_function]) or \
                           (curr_scaler_priority == best_scaler_priority and
                            distance_priority[dist_name] == distance_priority[self.best_distance_function] and
                                k < self.best_k):
                            self.best_k = k
                            self.best_distance_function = dist_name
                            self.best_scaler = scaler_name
                            self.best_model = model


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        if len(features) == 0:  # Check if empty using len
            return features

        # Convert to numpy array
        features = np.array(features, dtype=float)
        normalized = []

        for feature_vector in features:
            # Calculate the L2 normalization of the vector
            norm = np.linalg.norm(feature_vector)

            # If zero vector, keep it as is
            if norm == 0:
                normalized.append(feature_vector)
            else:
                # Normalize by dividing by the normalization
                normalized.append(feature_vector / norm)

        return np.array(normalized).tolist()


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
                For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
                This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
                The minimum value of this feature is thus min=-1, while the maximum value is max=2.
                So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
                leading to 1, 0, and 0.333333.
                If max happens to be same as min, set all new values to be zero for this feature.
                (For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        if len(features) == 0:
            return features

        # Convert to numpy array
        features = np.array(features, dtype=float)

        # Calculate the min and max for each feature columnwise
        min_values = np.min(features, axis=0)
        max_values = np.max(features, axis=0)

        # Avoid divison by zero: if max=min, set scaled values to 0
        scaled = np.zeros_like(features, dtype=float)

        # Scale each feature
        for j in range(features.shape[1]):
            denominator = max_values[j] - min_values[j]
            if denominator != 0:  # Check if max != min
                scaled[:, j] = (features[:, j] - min_values[j]) / denominator
            # else: leave as 0 when max equals min

        return scaled.tolist()
