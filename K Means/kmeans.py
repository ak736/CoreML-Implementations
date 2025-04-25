import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n)  # this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
    # according to some distribution, first generate a random number between 0 and
    # 1 using generator.rand(), then find the the smallest index n so that the
    # cumulative probability from example 1 to example n is larger than r.
    #############################################################################
    centers = [p]
    for _ in range(1, n_cluster):
        # Calculate squared distances from each point to the nearest existing center
        min_squared_distances = np.zeros(n)
        for i in range(n):
            min_dist = float('inf')
            for c in centers:
                dist = np.sum((x[i] - x[c])**2)
                if dist < min_dist:
                    min_dist = dist
            min_squared_distances[i] = min_dist

        # Convert distances to probabilities
        sum_distances = np.sum(min_squared_distances)
        if sum_distances == 0:
            # If all points are already centers or duplicates, choose randomly from remaining points
            remaining = list(set(range(n)) - set(centers))
            if remaining:
                new_center = generator.choice(remaining)
            else:
                new_center = generator.randint(0, n)
        else:
            probs = min_squared_distances / sum_distances

            # Choose the next center based on weighted probability
            r = generator.rand()
            cumulative_probs = np.cumsum(probs)

            # Find the smallest index i such that cumulative_probs[i] >= r
            new_center = 0
            for i in range(n):
                if r <= cumulative_probs[i]:
                    new_center = i
                    break

        centers.append(new_center)

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################

        centroids = x[self.centers].copy()
        membership = np.zeros(N, dtype=int)

        # Track convergence
        old_objective = float('inf')

        # Main loop
        for iter_count in range(self.max_iter):
            # Assign points to closest centroids
            distances = np.zeros((N, self.n_cluster))
            for k in range(self.n_cluster):
                distances[:, k] = np.sum((x - centroids[k])**2, axis=1)

            new_membership = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_cluster):
                points_in_cluster = x[new_membership == k]
                if len(points_in_cluster) > 0:
                    new_centroids[k] = np.mean(points_in_cluster, axis=0)
                else:
                    # If a cluster is empty, keep the old centroid
                    new_centroids[k] = centroids[k]

            # Calculate objective function (K-means distortion)
            objective = 0
            for i in range(N):
                objective += np.sum((x[i] -
                                    new_centroids[new_membership[i]])**2)

            # Check for convergence
            if abs(old_objective - objective) < self.e * old_objective:
                membership = new_membership
                centroids = new_centroids
                return centroids, membership, iter_count + 1

            # Update for next iteration
            membership = new_membership
            centroids = new_centroids
            old_objective = objective

        return centroids, membership, self.max_iter


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented,
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################

        # 1. Obtain centroids using KMeans
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, membership, _ = kmeans.fit(x, centroid_func)

        # 2. Assign labels to centroids by majority voting
        centroid_labels = np.zeros(self.n_cluster, dtype=y.dtype)
        for k in range(self.n_cluster):
            points_in_cluster = np.where(membership == k)[0]
            if len(points_in_cluster) > 0:
                # Get the labels of points in this cluster
                cluster_labels = y[points_in_cluster]
                # Find the most common label
                unique_labels, counts = np.unique(
                    cluster_labels, return_counts=True)
                centroid_labels[k] = unique_labels[np.argmax(counts)]

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################

        # For each test point, find the nearest centroid
        predicted_labels = np.zeros(N, dtype=self.centroid_labels.dtype)

        for i in range(N):
            # Calculate distances to all centroids
            distances = np.zeros(self.n_cluster)
            for k in range(self.n_cluster):
                distances[k] = np.sum((x[i] - self.centroids[k])**2)

            # Find the closest centroid
            closest_centroid = np.argmin(distances)

            # Assign the label of the closest centroid
            predicted_labels[i] = self.centroid_labels[closest_centroid]

        return predicted_labels


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################

    # Get image dimensions
    height, width, channels = image.shape

    # Reshape the image to a 2D array where each row is a pixel (RGB triplet)
    pixels = image.reshape(-1, channels)

    # Initialize the quantized image
    quantized_pixels = np.zeros_like(pixels)

    # For each pixel, find the nearest code vector
    for i in range(len(pixels)):
        min_distance = float('inf')
        nearest_idx = 0

        # Compare with all code vectors
        for j in range(len(code_vectors)):
            distance = np.sum((pixels[i] - code_vectors[j])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = j

        # Replace pixel with the nearest code vector
        quantized_pixels[i] = code_vectors[nearest_idx]

    # Reshape back to original image dimensions
    return quantized_pixels.reshape(height, width, channels)
