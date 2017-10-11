import sklearn.neural_netorks
import numpy as np


class AMLPClassifier(sklearn):

    def __init__(self, hidden_layer_size_ranges, *args, **kwargs):
        hidden_layer_sizes = tuple(
            size_range[0] for size_range in hidden_layer_size_ranges
        )

        self.hidden_layer_size_ranges = hidden_layer_size_ranges

        self.max_hidden_layer_sizes = tuple(
            size_range[-1] for size_range in hidden_layer_size_ranges
        )
        super(AMLPClassifier, self).__init__(hidden_layer_sizes, *args, **kwargs)

    def adapt_layer(self, layer_idx):
        # TODO: adapt layer

        pass

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
            result = super(AMLPClassifier, self).fit(X, y)
