import json
from .util import load_json
class Analyser:
    def __init__(self, path_correlation: str):
        self.corr = load_json(path_correlation)

        self.sorted_corr = list(zip(self.corr.keys(), self.corr.values()))
        self.sorted_corr.sort(key=lambda x: x[1], reverse=True)

    def corr(self, a, b):
        return self.corr[a + '_' + b]

    def max_n_corr(self, n):
        return self.sorted_corr[:n]

    def min_n_corr(self, n):
        return self.sorted_corr[-n:]
