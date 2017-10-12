import sklearn.neural_network
import numpy as np


class AMLPClassifier(sklearn.neural_network.MLPClassifier):

    def __init__(
            self, hidden_layer_size_ranges, activation='relu', solver='adam',
            alpha=0.0001, batch_size='auto', learning_rate='constant',
            learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
            random_state=None, tol=0.0001, verbose=False, warm_start=True,
            momentum=0.9, nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08):

        hidden_layer_sizes = tuple(
            size_range[0] for size_range in hidden_layer_size_ranges
        )

        self.hidden_layer_size_ranges = hidden_layer_size_ranges

        self.max_hidden_layer_sizes = tuple(
            size_range[-1] for size_range in hidden_layer_size_ranges
        )
        super(AMLPClassifier, self).__init__(
            hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate,
            learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose,
            warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction,
            beta_1, beta_2, epsilon)

    def adapt_layer(self, layer_idx):
        self.hidden_layer_sizes = tuple(
            size + 1 if idx == layer_idx else size
            for idx, size in enumerate(self.hidden_layer_sizes)
        )
        self.coefs_[layer_idx] = np.hstack(
            [
                self.coefs_[layer_idx],
                np.random.randn(self.coefs_[layer_idx].shape[0], 1)
            ]
        )
        self.coefs_[layer_idx + 1] = np.vstack(
            [
                self.coefs_[layer_idx + 1],
                np.random.randn(1, self.coefs_[layer_idx + 1].shape[1])
            ]
        )
        self.intercepts_[layer_idx] = np.hstack(
            [
                self.intercepts_[layer_idx],
                np.random.randn(1)
            ]
        )

    def can_adapt_layer(self, idx):
        return self.hidden_layer_sizes[idx] < self.max_hidden_layer_sizes[idx]

    def adapt(self):
        for layer_idx in range(len(self.hidden_layer_sizes)):
            self.adapt_layer(layer_idx)

    def can_adapt(self):
        return any(self.can_adapt_layer(idx)
                   for idx in range(len(self.hidden_layer_size_ranges)))

    def fit(self, X, y):
        result = super(AMLPClassifier, self).fit(X, y)
        while self.can_adapt():
            self.adapt()
            result = super(AMLPClassifier, self).fit(X, y)
        return result

if __name__ == "__main__":
    model = AMLPClassifier(
        hidden_layer_size_ranges=((5, 100), (2, 100)),
        verbose=True
    )
    X = np.random.randn(10000, 2)
    y = np.ravel(np.dot(X * X, np.ones((2, 1))) > 1)
    print model.fit(X, y)

