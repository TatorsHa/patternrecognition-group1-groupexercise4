import numpy as np
import os
import math

from src.data_formatting import load_features_from_tsv
from src.dtw import to_feature_vectors, dtw_distance


class TresholdClassifier:
    def __init__(self, genuine_signatures_folder: np.array, normalized=True):
        # computed by func computing_treshold on main
        # (mean 0 + mean 1) / 2 = 52914 not so efficient
        # Mean 1 ~ 30000 -> 35000 more balanced
        self.treshold = 300 if normalized else 35000
        self.normalize = normalized
        
        self.genuine_signatures = self.load_signatures(genuine_signatures_folder)

    def load_signatures(self, folder_path) -> dict:
        signatures = {}
        for f in os.listdir(folder_path):
            author_key = self.get_author(f)

            if author_key not in signatures:
                signatures[author_key] = []

            feature = load_features_from_tsv(os.path.join(folder_path, f), normalize=self.normalize)
            vectorized_feature = to_feature_vectors(feature)
            signatures[author_key].append(vectorized_feature)

        return signatures

    def get_author(self, filename):
        return filename[:3]

    def get_signatures(self, author=None):
        if author is None:
            return self.genuine_signatures
        return self.genuine_signatures[author]

    def compute_distances(self, candidate_filename, candidate_folder):
        author = self.get_author(candidate_filename)
        candidate_feature = load_features_from_tsv(
            os.path.join(candidate_folder, candidate_filename),
            normalize=self.normalize
        )
        vectorized_candidate = to_feature_vectors(candidate_feature)

        genuine_signatures = self.get_signatures(author)

        distances = []
        for sign in genuine_signatures:
            distances.append(dtw_distance(sign, vectorized_candidate))

        return distances

    def predict(self, candidate_filename, candidate_folder):
        distances = self.compute_distances(candidate_filename, candidate_folder)

        if min(distances) > self.treshold:
            return 0

        return 1
