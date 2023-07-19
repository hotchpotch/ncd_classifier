from __future__ import annotations
from typing import Callable
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from collections import defaultdict
import operator
from joblib import Parallel, delayed
from .compressors import COMPRESSORS


def default_concatenate_fn(
    data1: str | list[int], data2: str | list[int]
) -> str | list[int]:
    if isinstance(data1, str) and isinstance(data2, str):
        return data1 + " " + data2
    elif isinstance(data1, list) and isinstance(data2, list):
        return data1 + data2
    else:
        raise ValueError("data1 and data2 must be the same type.")


def compute_normalized_distance(len1: int, len2: int, combined_len: int) -> float:
    """
    Calculates the normalized compression distance between two pieces of data.
    """
    return (combined_len - min(len1, len2)) / max(len1, len2)


class NPCClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier based on the Normalized Compression Distance (NCD).
    """

    def __init__(
        self,
        concatenate_fn: Callable[
            [str | list[int], str | list[int]], str | list[int]
        ] = default_concatenate_fn,
        compute_distance: Callable[
            [int, int, int], float
        ] = compute_normalized_distance,
        compress_len_fn: Callable[[str | list[int]], int] = COMPRESSORS["zlib"],
        k: int = 3,
        n_jobs: int = 1,
    ):
        """
        Initializes the NPCClassifier.
        """
        self.concatenate_fn = concatenate_fn
        self.compute_distance = compute_distance
        self.compress_len_fn = compress_len_fn
        self.k = k
        self.n_jobs = n_jobs
        self.train_data = []
        self.train_labels = []
        self.compressed_train_data = []

    def fit(self, X: list[str | list[int]], y: list[int]) -> NPCClassifier:
        """
        Fits the model using the training data.
        """
        self.train_data = X
        self.train_labels = y
        compressed_train_data = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compress_len_fn)(x) for x in X
        )
        self.compressed_train_data = list(compressed_train_data)
        return self

    def predict(self, X: list[str | list[int]]) -> list[int]:
        """
        Predicts the labels of the given data.
        """
        predicted_labels = Parallel(n_jobs=self.n_jobs)(
            delayed(self.predict_single)(x) for x in X
        )
        return predicted_labels

    def predict_single(self, data: str | list[int]) -> int:
        """
        Predicts the label of a single instance.
        """
        distances = self.calculate_distances_to_train_data(self.train_data, data)
        sorted_indices = np.argsort(np.array(distances))
        label_counts = defaultdict(int)
        for label in self.train_labels:
            label_counts[label] += 1
        total_label_counts = sum(label_counts.values())
        for label in label_counts:
            label_counts[label] /= total_label_counts
            label_counts[label] = 1 / label_counts[label]
        nearest_label_counts = defaultdict(int)
        for j in range(self.k):
            nearest_label = self.train_labels[sorted_indices[j]]
            nearest_label_counts[nearest_label] += 1
        for label in label_counts:
            # Taking into account the occurrence rate
            nearest_label_counts[label] *= label_counts[label]

        sorted_label_counts = sorted(
            nearest_label_counts.items(), key=operator.itemgetter(1), reverse=True
        )
        most_frequent_label = sorted_label_counts[0][0]
        return most_frequent_label

    def calculate_distances_to_train_data(
        self, train_data: list[str | list[int]], test_data: str | list[int]
    ) -> list[float]:
        """
        Calculates the distances from a test instance to all training instances.
        """
        distances = []
        test_data_compressed_len = self.compress_len_fn(test_data)
        for j, train_data in enumerate(train_data):
            train_data_compressed_len = self.compressed_train_data[j]
            combined_data_compressed_len = self.compress_len_fn(
                self.concatenate_fn(test_data, train_data)
            )
            distance = self.compute_distance(
                test_data_compressed_len,
                train_data_compressed_len,
                combined_data_compressed_len,
            )
            distances.append(distance)
        return distances
