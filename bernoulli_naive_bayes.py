import numpy as np


class BernoulliNaiveBayes:

    def __init__(self):
        self.category_list = {}
        self.theta_matrix = np.array([])
        self.num_classes = 0
        self.num_features = 0
        self.keys = []
        self.values = []

    def _get_theta(self, X, y):
        # number of categories x number of features
        theta_matrix = np.zeros((len(self.category_list), X.shape[1]))

        # get all unique keys
        self.keys = list(self.category_list.keys())
        self.values = list(self.category_list.values())

        for feature_row, category in zip(X, y):
            for i, key in enumerate(self.keys):
                if category is key:
                    feature_row = feature_row.toarray()[0, :]
                    for j, feature in enumerate(feature_row):
                        if feature > 0:
                            theta_matrix[i][j] += 1

        # the matrix contains number of examples where feature_j > 0 and y corresponds to its class
        for i in range(len(self.values)):
            theta_matrix[i, :] = (theta_matrix[i, :] + 1) / (self.values[i] + 2)
        return theta_matrix

    def _get_marginal_probability(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    def fit(self, X, y):
        self.category_list = self._get_marginal_probability(y)
        self.theta_matrix = self._get_theta(X, y)
        self.num_classes = len(y)
        self.num_features = X.shape[1]

    def predict(self, X_test):
        y_output_list = []
        for feature_row in X_test:
            feature_row = feature_row.toarray()[0, :]
            class_prob = []
            for k in range(len(self.values)):
                feature_likelihood = 0
                for j in range(self.num_features):
                    if feature_row[j] > 0 and self.theta_matrix[k][j] != 0:
                        feature_likelihood += np.log(self.theta_matrix[k][j])
                    elif self.theta_matrix[k][j] != 1:
                        feature_likelihood += np.log(1-self.theta_matrix[k][j])
                class_prob.append(feature_likelihood + np.log(self.values[k]/self.num_classes))
            index = class_prob.index(max(class_prob))
            y_output_list.append(self.keys[index])
        return y_output_list
