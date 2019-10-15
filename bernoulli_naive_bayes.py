import numpy as np


class BernoulliNaiveBayes:

    def __init__(self, alpha=1.0):
        """Initialize instance of BernoulliNaiveBayes class
        :param alpha (default = 1.0): Laplace smoothing parameter
        """
        self.keys = []
        self.values = []
        self.num_categories = 0
        self.theta_matrix = np.array([])
        self.num_classes = 0
        self.num_features = 0
        self.alpha = alpha

    def _get_theta(self, X, y):
        """Calculate parameters needed for compute P(x_m|y)
        :param X: shape = [n, m]
                  matrix of training examples where n is the number of samples and m is the number of features
        :param y: shape = [n]
                  target values for training examples
        :return: shape = [num_categories, m]
                 matrix of theta values where num_categories is the number of categories and m is the number of features
                 num_categories = 20 in mini-project2
        """
        # initialize the matrix that contains all theta values
        theta_matrix = np.zeros((self.num_categories, self.num_features))

        for feature_row, category in zip(X, y):
            n = self.keys.index(category)
            feature_row = feature_row.toarray()[0, :]
            feature_row[feature_row > 0] = 1
            theta_matrix[n, :] += feature_row

        val = np.reshape(self.values, (self.num_categories, 1))
        theta_matrix = (theta_matrix + self.alpha) / (val + self.alpha * self.num_categories)
        return theta_matrix

    def _get_category_occurrences(self, y):
        """Determine the categories and each category's number of occurrences in the target values
        :param y: shape = [n]
                  target values for training examples
        :return: dictionary where keys are the unique target values and values are the numbers of occurrences
                 of each unique target value
        """
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    def fit(self, X, y):
        """Fit Bernoulli Naive Bayes classifier according to X, y
        :param X: shape = [n, m]
                  matrix of training examples where n is the number of samples and m is the number of features
        :param y: shape = [n]
                  target values for training examples
        """
        category_list = self._get_category_occurrences(y)
        self.keys = list(category_list.keys())      # keys contains categories
        self.values = list(category_list.values())  # values contains numbers of occurrences of each category
        self.num_categories = len(self.keys)        # number of categories

        self.num_classes = X.shape[0]               # number of samples
        self.num_features = X.shape[1]              # number of features

        self.theta_matrix = self._get_theta(X, y)

    def predict(self, X_test):
        """Perform classification on an array of test vectors X_test
        :param X_test: shape = [n_test, m]
                       matrix of testing examples where n_test is the number of samples and m is the number of features
        :return: shape = [n_test]
                 predicted target values for X_test
        """
        y_output_list = []

        theta = np.transpose(self.theta_matrix)
        log_theta = np.log(theta)
        log_comp_theta = np.log(1-theta)

        marginal_prob = np.log(np.asarray(self.values) / self.num_classes)

        for feature_row in X_test:
            feature_row = feature_row.toarray()[0, :]
            feature_row[feature_row > 0] = 1
            feature_row = np.reshape(feature_row, (1, len(feature_row)))

            class_prob = np.dot(feature_row, log_theta) + np.dot((1 - feature_row), log_comp_theta)
            class_prob += marginal_prob

            index = np.argmax(class_prob)
            y_output_list.append((self.keys[index]))
        return y_output_list
